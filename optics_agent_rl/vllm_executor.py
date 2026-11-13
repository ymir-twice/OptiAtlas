# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-host FileStore backport for the pinned vLLM 0.19.1 executor.

Adapted from vLLM's MultiprocExecutor._init_executor and the rendezvous fix in
https://github.com/vllm-project/vllm/pull/50999 (aoshen02 / Parsa Mazaheri).
The copied initialization body differs only in its rendezvous URI assignment;
all worker spawning, queues, monitoring and failure cleanup stay upstream.
Use the official distributed_executor_backend class hook. No global patching.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import suppress
import weakref

from vllm import __version__ as VLLM_VERSION
from vllm.v1.executor import multiproc_executor as _mp
from vllm.v1.executor.multiproc_executor import (
    FailureCallback, FutureWrapper, Handle, MessageQueue, MultiprocExecutor,
    UnreadyWorkerProcHandle, WorkerProc, current_platform, envs, get_ip,
    get_mp_context, logger, set_multiprocessing_worker_envs,
)

from .vllm_rendezvous import LocalFileRendezvous, is_local_file_store_config


class FileStoreMultiprocExecutor(MultiprocExecutor):
    """Use FileStore only for the validated NVIDIA single-node TP executor.

    Other topologies retain upstream initialization. This fallback does not
    bypass vLLM's earlier restrictions on custom backends with PP or multi-node.
    """

    def __init__(self, vllm_config, monitor_workers=True):
        self._file_rendezvous = None
        self._rendezvous_worker_processes = []
        if current_platform.is_cuda() and is_local_file_store_config(vllm_config):
            if VLLM_VERSION != "0.19.1":
                raise RuntimeError("FileStoreMultiprocExecutor requires pinned vLLM 0.19.1")
            from .vllm_executor_contract import validate_pinned_source
            validate_pinned_source(_mp.__file__)
            self._file_rendezvous = LocalFileRendezvous()
        try:
            super().__init__(vllm_config, monitor_workers=monitor_workers)
        except BaseException:
            # The copied initialization cleans partially spawned workers first.
            # Also release queues and the private directory if construction fails.
            with suppress(Exception):
                self.shutdown()
            raise

    def _init_executor(self):
        if self._file_rendezvous is None:
            return super()._init_executor()
        logger.info("Using single-node FileStore rendezvous: %s", self._file_rendezvous.init_method)
        return self._init_file_executor()

    def _ensure_worker_termination(self, worker_procs):
        # Keep ownership even when initialization fails before self.workers is
        # assigned. The upstream body already calls this method on that path.
        self._rendezvous_worker_processes = list(worker_procs)
        return super()._ensure_worker_termination(worker_procs)

    def shutdown(self):
        stopped = False
        try:
            super().shutdown()
            stopped = True
        finally:
            if rendezvous := getattr(self, "_file_rendezvous", None):
                # Upstream marks shutting_down before closing death_writer. If
                # closing a pipe raises, a second shutdown skips termination.
                # Ready handles must still guard the store even when our
                # _ensure_worker_termination hook was never reached.
                workers = list(getattr(self, "_rendezvous_worker_processes", []))
                workers.extend(handle.proc for handle in (getattr(self, "workers", None) or []))
                if not stopped or any(process.is_alive() for process in workers):
                    logger.warning("Retaining owned FileStore until all worker processes terminate")
                else:
                    try:
                        rendezvous.close()
                    except Exception:
                        logger.warning("Could not clean owned FileStore rendezvous", exc_info=True)

    def _init_file_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()
        assert self.world_size == tp_size * pp_size * pcp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}). "
        )

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # The upstream local TCP rendezvous is replaced with an owned FileStore.
        distributed_init_method = self._file_rendezvous.init_method
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0:
            # For leader node within each dp rank,
            # each dp will have its own leader multiproc executor.
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip()
            logger.info(
                "DP group leader: node_rank=%d, node_rank_within_dp=%d, "
                "master_addr=%s, mq_connect_ip=%s (local), "
                "world_size=%d, local_world_size=%d",
                self.parallel_config.node_rank,
                self.parallel_config.node_rank_within_dp,
                self.parallel_config.master_addr,
                mq_connect_ip,
                self.world_size,
                self.local_world_size,
            )
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            )
            # When using fork, keep track of socket file descriptors that are
            # inherited by the worker, so that we can close them in subsequent
            # workers
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                )
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

