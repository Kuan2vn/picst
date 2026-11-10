def add_underwater_label_args(parser):
    parser.add_argument(
        "--seafloor_ignore",
        "--seafloor-ignore",
        dest="seafloor_ignore",
        action="store_true",
        help="Use the legacy 7-class SUIM protocol by mapping sea-floor/rocks to ignore_label.",
    )
    parser.add_argument(
        "--suim-legacy-color-map",
        action="store_true",
        help=(
            "Use the old repo SUIM color map for exact legacy result recovery. "
            "This also enables --seafloor_ignore and sets MODEL.NUM_CLASSES=7."
        ),
    )


def apply_underwater_label_args(cfg, args):
    if getattr(args, "seafloor_ignore", False):
        cfg.DATASETS.SEAFLOOR_IGNORE = True
        cfg.MODEL.NUM_CLASSES = 7
    if getattr(args, "suim_legacy_color_map", False):
        cfg.DATASETS.SEAFLOOR_IGNORE = True
        cfg.DATASETS.SUIM_LEGACY_COLOR_MAP = True
        cfg.MODEL.NUM_CLASSES = 7
