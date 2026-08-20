"""単一動画選定applicationの実行境界."""

import logging

from ..models.video_selection_request import VideoSelectionRequest
from ..services.single_video_selector import SingleVideoSelector

logger = logging.getLogger(__name__)


def run_video_application(request: VideoSelectionRequest) -> None:
    """単一動画からブログ掲載用画像を選定する."""
    if request.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        SingleVideoSelector(request).run()
    except KeyboardInterrupt as error:
        logger.info("中断されました。同じコマンドで保存済み進捗から再開できます。")
        raise SystemExit(130) from error
    except Exception as error:
        logger.error("画像選定に失敗しました: %s: %s", type(error).__name__, error)
        raise SystemExit(1) from error
