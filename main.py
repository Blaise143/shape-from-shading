import argparse
from src.run import run
from src.data_load import ShapeFromShadingDataset
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Shape-from-Shading on a dataset of images + masks."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        # default="GT_data4/Cue1",
        required=True,
        help="Image directory",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        required=True,
        # default="GT_data4/Masks",
        help="Directory with mask images.",
    )

    parser.add_argument(
        "--out-path",
        type=str,
        default="out",
        help="Base output directory.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="If name of target is passed, sfs runs only on the specified object, nothing else",
    )

    parser.add_argument(
        "--num-iters",
        type=int,
        default=2000,
        help="Number of optimization iterations.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate.",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=1.0,
        help="Weight for smoothness regularization.",
    )
    parser.add_argument(
        "--light-direction",
        nargs=3,
        type=float,
        metavar=("LX", "LY", "LZ"),
        default=[0.0, 0.0, 1.0],
    )

    parser.add_argument(
        "--no-optimize-light",
        dest="optimize_light",
        action="store_false",
        help="Disable light optimization (default: optimize).",
    )
    parser.add_argument(
        "--optimize-light",
        dest="optimize_light",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(optimize_light=True)

    parser.add_argument(
        "--optimize-albedo",
        action="store_true",
        help="Enable joint optimization of albedo.",
    )
    parser.add_argument(
        "--init-kind",
        type=str,
        choices=["init_grads", "zeroes", "ones", "sphere"],
        default="init_grads",
        help="Initialization type for normals.",
    )
    parser.add_argument(
        "--parametrization-type",
        type=str,
        choices=["cartesian", "stereographic"],
        default="stereographic",
        help="Normal parametrization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ShapeFromShadingDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
    )
    light_dir_tensor = torch.tensor(args.light_direction, dtype=torch.float32)

    run(
        dataset=dataset,
        out_path=args.out_path,
        target=args.target,
        num_iters=args.num_iters,
        lr=args.lr,
        lambda_smooth=args.lambda_smooth,
        optimize_light=args.optimize_light,
        light_direction=light_dir_tensor,
        optimize_albedo=args.optimize_albedo,
        init_kind=args.init_kind,
        parametrization_type=args.parametrization_type,
    )


if __name__ == "__main__":
    main()
