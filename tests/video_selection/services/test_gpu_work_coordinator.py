from threading import Event, Lock, Thread

from src.video_selection.services.gpu_work_coordinator import GpuWorkCoordinator


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
