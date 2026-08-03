from threading import Event, Lock, Thread

from src.video_selection.services.gpu_work_coordinator import GpuWorkCoordinator


def test_gpu_work_coordinator_limits_parallel_vision_work() -> None:
    """Vision workが設定上限まで並列化され超過分は待機させられること。

    Arrange:
        - Vision同時実行上限2のcoordinatorと三つのthreadが用意される
    Act:
        - 三つのVision workが同時に要求される
    Assert:
        - 最大2件が重なり、解放後に三つすべて完了すること
    """
    # Arrange
    coordinator = GpuWorkCoordinator(max_parallel_requests=2)
    state_lock = Lock()
    two_started = Event()
    release = Event()
    active_count = 0
    maximum_active_count = 0
    completed: list[int] = []

    def work(index: int) -> None:
        nonlocal active_count, maximum_active_count
        with state_lock:
            active_count += 1
            maximum_active_count = max(maximum_active_count, active_count)
            if active_count == 2:
                two_started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("Vision workを解放できませんでした")
        with state_lock:
            active_count -= 1
            completed.append(index)

    threads = tuple(
        Thread(
            target=lambda index=index: coordinator.run(
                "vision_inference",
                lambda: work(index),
            )
        )
        for index in range(3)
    )

    # Act
    for thread in threads:
        thread.start()
    if not two_started.wait(timeout=1.0):
        raise RuntimeError("二つのVision workが並列に開始されませんでした")
    release.set()
    for thread in threads:
        thread.join(timeout=1.0)

    # Assert
    assert maximum_active_count == 2
    assert sorted(completed) == [0, 1, 2]
    assert not any(thread.is_alive() for thread in threads)


def test_gpu_work_coordinator_waits_for_active_vision_before_speech() -> None:
    """実行中のVisionがすべて終わるまでSTTが開始されないこと。

    Arrange:
        - 同時実行中の二つのVision workと、後から要求されるSTTが用意される
    Act:
        - Vision workの実行中にSTTが要求され、その後Visionが解放される
    Assert:
        - STTは二つのVision完了後に単独で開始されること
    """
    # Arrange
    coordinator = GpuWorkCoordinator(max_parallel_requests=2)
    state_lock = Lock()
    visions_started = Event()
    release_visions = Event()
    speech_attempted = Event()
    speech_started = Event()
    active_count = 0
    maximum_active_count = 0
    completion_order: list[str] = []

    def vision(name: str) -> None:
        nonlocal active_count, maximum_active_count
        with state_lock:
            active_count += 1
            maximum_active_count = max(maximum_active_count, active_count)
            if active_count == 2:
                visions_started.set()
        if not release_visions.wait(timeout=1.0):
            raise RuntimeError("Vision workを解放できませんでした")
        with state_lock:
            active_count -= 1
            completion_order.append(name)

    def speech() -> None:
        nonlocal active_count, maximum_active_count
        speech_attempted.set()

        def operation() -> None:
            nonlocal active_count, maximum_active_count
            with state_lock:
                active_count += 1
                maximum_active_count = max(maximum_active_count, active_count)
                speech_started.set()
                active_count -= 1
                completion_order.append("speech")

        coordinator.run("speech_to_text", operation)

    vision_threads = tuple(
        Thread(
            target=lambda name=name: coordinator.run(
                "vision_inference",
                lambda: vision(name),
            )
        )
        for name in ("vision-1", "vision-2")
    )
    speech_thread = Thread(target=speech)

    # Act
    for thread in vision_threads:
        thread.start()
    if not visions_started.wait(timeout=1.0):
        raise RuntimeError("二つのVision workが開始されませんでした")
    speech_thread.start()
    if not speech_attempted.wait(timeout=1.0):
        raise RuntimeError("STTが要求されませんでした")
    speech_started_while_vision_active = speech_started.is_set()
    release_visions.set()
    for thread in (*vision_threads, speech_thread):
        thread.join(timeout=1.0)

    # Assert
    assert speech_started_while_vision_active is False
    assert maximum_active_count == 2
    assert completion_order[-1] == "speech"
    assert not any(thread.is_alive() for thread in (*vision_threads, speech_thread))


def test_gpu_work_coordinator_serializes_speech_and_vision_work() -> None:
    """STTとOllama相当workが同じprocess内で重ならず実行されること。

    Arrange:
        - 同じcoordinatorを共有する二つのthreadと同期eventが用意される
    Act:
        - speech-to-text実行中にvision inferenceが要求される
    Assert:
        - active GPU workの最大件数が1で両方完了すること
    """
    # Arrange
    coordinator = GpuWorkCoordinator()
    state_lock = Lock()
    first_started = Event()
    second_attempted = Event()
    release_first = Event()
    active_count = 0
    maximum_active_count = 0
    completed: list[str] = []

    def work(name: str, *, block: bool) -> str:
        nonlocal active_count, maximum_active_count
        with state_lock:
            active_count += 1
            maximum_active_count = max(maximum_active_count, active_count)
        if block:
            first_started.set()
            if not release_first.wait(timeout=1.0):
                msg = "最初のGPU workを解放できませんでした"
                raise RuntimeError(msg)
        with state_lock:
            active_count -= 1
            completed.append(name)
        return name

    def run_speech() -> None:
        coordinator.run("speech_to_text", lambda: work("speech", block=True))

    def run_vision() -> None:
        second_attempted.set()
        coordinator.run("vision_inference", lambda: work("vision", block=False))

    first = Thread(target=run_speech)
    second = Thread(target=run_vision)

    # Act
    first.start()
    if not first_started.wait(timeout=1.0):
        msg = "最初のGPU workが開始されませんでした"
        raise RuntimeError(msg)
    second.start()
    if not second_attempted.wait(timeout=1.0):
        msg = "二つ目のGPU workが要求されませんでした"
        raise RuntimeError(msg)
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    # Assert
    assert (maximum_active_count, completed, first.is_alive(), second.is_alive()) == (
        1,
        ["speech", "vision"],
        False,
        False,
    )
