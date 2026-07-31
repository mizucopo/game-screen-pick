"""解決済みmodelからSpeechRuntimeを構築するfactory port。"""

from collections.abc import Callable

from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_model import ResolvedModel
from .speech_runtime import SpeechRuntime

SpeechRuntimeFactory = Callable[
    [ResolvedModel, EffectiveConfiguration],
    SpeechRuntime,
]
