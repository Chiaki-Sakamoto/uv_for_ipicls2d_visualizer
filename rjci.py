#!/usr/bin/python3

# pyright: reportMissingTypeStubs=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false

from ast import Not
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from scipy.constants import pi
from scipy.constants import epsilon_0
from scipy.constants import elementary_charge
from scipy.constants import electron_mass
from scipy.constants import speed_of_light
import os
import sys
import argparse
import f90nml
import numpy as np
import matplotlib

matplotlib.use(backend="Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator


class NameList:
    def __init__(self, namelist: str) -> None:
        try:
            with open(namelist) as file:
                config: dict[str, int | float | bool] = f90nml.read(file)
            _ = sys.stdout.write(f"namelist{namelist} を読み込みました. ")
        except FileNotFoundError as error:
            raise RuntimeError("設定ファイルが必須ですが見つかりません") from error
        self.Nx: int = config["geom"]["Nx"]
        self.Ny: int = config["geom"]["Ny"]
        self.NV: int = config["geom"]["NV"]
        self.NM: int = config["geom"]["NM"]
        self.boost_opt: bool = config["geom"]["boost_opt"]
        self.nst_boost: int = config["geom"]["nst_boost"]
        self.csim: float = config["geom"]["c"]
        self.system_lx: int = config["geom"]["system_lx"]
        self.system_ly: int = config["geom"]["system_ly"]
        self.wavelength: float = config["wave"]["wavelength"]
        self.ndav: int = config["diag"]["ndav"]
        self.Nx_d: int = config["diag"]["Nx_d"]
        self.Ny_d: int = config["diag"]["Ny_d"]
        self.int_snap: int = config["diag"]["int_snap"]
        self.ow: float = config["wave"]["ow"]
        self.Exext: float = config["wave"]["Exext"]
        self.Ezext: float = config["wave"]["Ezext"]
        self.ionize_opt: bool = config["ionize"]["ionize_opt"]
        self.zin0: float = config["ionize"]["zin0"]
        self.field_ionize_opt: bool = config["ionize"]["field_ionize_opt"]
        self.col_ionize_opt: bool = config["ionize"]["col_ionize_opt"]
        self.impact_ionize_opt: bool = config["ionize"]["impact_ionize_opt"]
        self.grid_x: int = int(self.Nx // self.Nx_d) + 1
        self.grid_y: int = int(self.Ny // self.Ny_d) + 1
        self.omega_L: float = (2 * pi * speed_of_light) / (self.wavelength * 1.0e-6)
        self.n_cr: float = (
            (electron_mass + epsilon_0 + self.omega_L**2)
            / elementary_charge**2
            * 1.0e-6
        )
        self.n_pic: float = self.n_cr / self.ow**2
        self.T: float = 2 * pi / self.ow
        self.d_t: float = self.T / self.ndav
        return

    def __del__(self) -> None:
        return


class FigureConfig:
    def __init__(self, args: argparse.Namespace, namelist: NameList) -> None:
        try:
            if getattr(args, "x_direction", False):
                self.column: int = 0
                self.figureTitle: str = "Jx"
            elif getattr(args, "y_direction", False):
                self.column = 1
                self.figureTitle = "Jy"
            elif getattr(args, "z_direction", False):
                self.column = 2
                self.figureTitle = "Jz"
            else:
                self.column = 2
                self.figureTitle = "Jz"
        except AttributeError as error:
            raise AttributeError("引数 'x_direction' が指定されていません") from error
        if getattr(args, "line", False):
            self.lineFlag: bool = True
        else:
            self.lineFlag: bool = False
        if getattr(args, "x_min", False):
            self.x_min: float = args.x_min
        else:
            self.x_min: float = 0
        if getattr(args, "x_max", False):
            self.x_max: float = args.x_max
        else:
            self.x_max: float = 0
        if getattr(args, "y_min", False):
            self.y_min: float = args.y_min
        else:
            self.y_min: float = 0
        if getattr(args, "y_max", False):
            self.y_max: float = args.y_max
        else:
            self.y_max: float = 0
        if getattr(args, "color_map_z_min", False) or getattr(
            args, "color_map_z_max", False
        ):
            self.cz_min: float = args.color_map_z_min
            self.cz_max: float = args.color_map_z_max
        else:
            self.cz_min: float = 0
            self.cz_max: float = 0
        if getattr(args, "output_directory", False):
            self.output_directory: str = args.output_directory
        else:
            self.output_directory: str = "./"
        plt.rcParams["font.size"] = 20
        return

    def __del__(self) -> None:
        return


def parseArgument():
    parser = argparse.ArgumentParser(
        prog="rjci.py",
        usage="rjci.py [-n --namelist] namelist_file [-l --line] \
            [-x --x_min] [-X --x_max] [-y --y_min] [-Y --y_max] \
            [-cz --color_map_z_min] [-cZ --color_map_z_max]",
        description="""【概要】
        ipicls2d によって計算された電流密度瞬時値を 2 次元カラーマップで rjci_plot 内に保存し，そのアニメーションも作成する．
        -x/--x_min, -X/--x_max, -y/--y_min, -Y/--y_max の指定でx軸とy軸のプロット範囲を指定することができる．
        -cz/--color_map_z_min, -cZ/--color_map_z_max でマップのカラーバーの範囲(最小値，最大値)
        ただし，-cz と -cZ は同時に両方指定しなければならない．
        -l/--line でx軸上(y=0)でのラインプロット．csv dir にそのときのデータファイルを出力される．
        【注意】負の数はダブルコーテーションで囲み，かつマイナスの前に半角スペースを入れること．
        """,
        epilog="Example: python3 ./rjci.py -n ./namelist_file",
        add_help=True,
    )
    _ = parser.add_argument("-n", "--namelist", type=str, required=True)
    _ = parser.add_argument(
        "-dx", "--x_direction", help="x 方向の電流密度を可視化.", action="store_true"
    )
    _ = parser.add_argument(
        "-dy", "--y_direction", help="y 方向の電流密度を可視化.", action="store_true"
    )
    _ = parser.add_argument(
        "-dz", "--z_direction", help="z 方向の電流密度を可視化.", action="store_true"
    )
    _ = parser.add_argument(
        "-l",
        "--line",
        help="各時刻における y=0 での物理量をラインプロットする. ",
        action="store_true",
    )
    _ = parser.add_argument(
        "-x",
        "--x_min",
        help="プロットする範囲(x 軸の最小値)",
        type=float,
        action="store",
    )
    _ = parser.add_argument(
        "-X",
        "--x_max",
        help="プロットする範囲(x 軸の最大値)",
        type=float,
        action="store",
    )
    _ = parser.add_argument(
        "-y",
        "--y_min",
        help="プロットする範囲(y 軸の最小値)",
        type=float,
        action="store",
    )
    _ = parser.add_argument(
        "-cz",
        "--color_map_z_min",
        help="カラーバーの最小値",
        type=float,
        action="store",
    )
    _ = parser.add_argument(
        "-cZ",
        "--color_map_z_max",
        help="カラーバーの最大値",
        type=float,
        action="store",
    )
    _ = parser.add_argument(
        "-od",
        "--output_directory",
        help="出力先ディレクトリ",
        type=str,
        default="./",
    )
    return parser.parse_args()


def make_directory(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        print(f"ディレクトリ '{path}' を作成しました（または既に存在しています）")
    except PermissionError as error:
        raise RuntimeError(f"権限エラー: '{path}' を作成できません") from error
    except OSError as error:
        raise RuntimeError(
            f"OSエラー: '{path}' の作成に失敗しました。詳細: {error}"
        ) from error


def directory_exists(path: str) -> bool:
    return os.path.isdir(path)


def figureSetting(
    axes: Axes, parameters: NameList, figureConfig: FigureConfig, dataNum: int
) -> None:
    realTime: float = (
        dataNum
        * parameters.int_snap
        * parameters.wavelength
        * 1.0e-6
        / speed_of_light
        / parameters.ndav
        * 1.0e15
    )
    timeStep: int = dataNum * parameters.ndav

    # xaxis setting
    _ = axes.set_xlabel("x [$\\mathrm{\\mu m}$]")
    if figureConfig.x_min != 0 or figureConfig.x_max != 0:
        _ = axes.set_xlim(float(figureConfig.x_min), float(figureConfig.x_max))
        _ = axes.xaxis.set_major_locator(MultipleLocator(20))
    else:
        _ = axes.set_xlim()
        _ = axes.xaxis.set_major_locator(MultipleLocator(40))
    axes.tick_params(axis="x", pad=10)
    axes.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # yaxis setting
    if figureConfig.lineFlag:
        _ = axes.set_ylabel(f"{figureConfig.figureTitle} [ norm. ]")
    else:
        _ = axes.set_ylabel("y [$\\mathrm{\\mu m}$]")
    axes.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if figureConfig.y_min != 0 or figureConfig.y_max != 0:
        _ = axes.set_ylim(float(figureConfig.y_min), float(figureConfig.y_max))
    elif (
        figureConfig.y_min == 0
        and figureConfig.y_max == 0
        and not figureConfig.lineFlag
    ):
        _ = axes.set_ylim()
        _ = axes.yaxis.set_major_locator(MultipleLocator(40))
    axes.tick_params(axis="y", pad=10)
    axes.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # title setting
    _ = axes.set_title(figureConfig.figureTitle, size="medium", loc="center")
    _ = axes.set_title(
        f"{realTime:8.02f}" + " fsec=" + f"{timeStep}" + " timestep",
        size="small",
        loc="left",
    )
    _ = axes.set_title("file=" + format(dataNum) + "    ", size="small", loc="right")

    # tick setting
    _ = axes.tick_params(which="both", right="on", top="on")
    _ = axes.tick_params(which="major", direction="in", width=2, length=10, colors="k")
    _ = axes.tick_params(which="minor", direction="in", length=5, colors="k")
    return


def xyRangeSetting(
    plotData: NDArray[np.float64], parameters: NameList, flag: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if flag:
        x: NDArray[np.float64] = (
            np.linspace(
                -plotData.shape[1] / 2, plotData.shape[1] / 2, plotData.shape[1] + 1
            )
            * (parameters.wavelength / parameters.ndav)
            * parameters.Nx_d
        )
        y: NDArray[np.float64] = (
            np.linspace(
                plotData.shape[0] / 2, -plotData.shape[0] / 2, plotData.shape[0] + 1
            )
            * (parameters.wavelength / parameters.ndav)
            * parameters.Ny_d
        )
    else:
        x: NDArray[np.float64] = (
            np.linspace(
                -plotData.shape[1] / 2, plotData.shape[1] / 2, plotData.shape[1]
            )
            * (parameters.wavelength / parameters.ndav)
            * parameters.Nx_d
        )
        y: NDArray[np.float64] = (
            np.linspace(
                plotData.shape[0] / 2, -plotData.shape[0] / 2, plotData.shape[0]
            )
            * (parameters.wavelength / parameters.ndav)
            * parameters.Ny_d
        )
    return x, y


def linePlot(
    parameters: NameList, figureConfig: FigureConfig, filePath: str, dataNum: int
) -> None:
    plotData = np.zeros([parameters.grid_x, parameters.grid_y])
    data = np.loadtxt(filePath)
    f, ax = plt.subplots(figsize=(10, 6))

    for i in range(parameters.grid_y):
        for j in range(parameters.grid_x):
            plotData[i, j] = data[i * parameters.grid_x + j, figureConfig.column]
    x, _ = xyRangeSetting(plotData, parameters, False)
    _ = ax.plot(x, plotData[parameters.grid_y // 2, :])
    figureSetting(ax, parameters, figureConfig, dataNum)
    _ = ax.xaxis.set_major_locator(MultipleLocator(40))
    if figureConfig.y_min == 0 and figureConfig.y_max == 0:
        _ = ax.set_ylim(
            np.amin(plotData[parameters.grid_y // 2, :])
            - np.amin(plotData[parameters.grid_y // 2, :]) * 0.1,
            np.amax(plotData[parameters.grid_y // 2, :])
            + np.amax(plotData[parameters.grid_y // 2, :] * 0.1),
        )
    plt.tight_layout()
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    f.canvas.draw()
    _ = plt.savefig(
        "./rjci_plot/"
        + figureConfig.output_directory
        + figureConfig.figureTitle
        + "_rjciLine_"
        + "{0:05d}".format(dataNum)
        + ".png"
    )
    plt.close("all")
    del f, ax, data, plotData
    return


def colorScalePlot(
    parameters: NameList, figureConfig: FigureConfig, filePath: str, dataNum: int
) -> None:
    plotData = np.zeros([parameters.grid_x, parameters.grid_y])
    data = np.loadtxt(filePath)
    f, ax = plt.subplots(figsize=(10, 6))

    for i in range(parameters.grid_y):
        for j in range(parameters.grid_x):
            plotData[i, j] = data[i * parameters.grid_x + j, figureConfig.column]
    x, y = xyRangeSetting(plotData, parameters, True)

    divider = make_axes_locatable(ax)
    colorAxes = divider.append_axes("right", "5%", pad="2%")
    if figureConfig.cz_min != 0 or figureConfig.cz_max != 0:
        image = ax.pcolorfast(
            x,
            y,
            plotData,
            vmin=float(figureConfig.cz_min),
            vmax=float(figureConfig.cz_max),
            cmap="rainbow",
        )
    else:
        image = ax.pcolorfast(x, y, plotData, cmap="rainbow")
    colorBar = f.colorbar(image, cax=colorAxes)
    colorBar.set_label(figureConfig.figureTitle + " [norm.]", size=24)
    colorBar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    figureSetting(ax, parameters, figureConfig, dataNum)
    f.canvas.draw()
    plt.tight_layout()
    _ = plt.savefig(
        "./rjci_plot/"
        + figureConfig.output_directory
        + figureConfig.figureTitle
        + "_rjci_"
        + "{0:05d}".format(dataNum)
        + ".png"
    )
    plt.close("all")
    del colorBar, f, ax, image, data, plotData
    return


def main() -> None:
    try:
        args = parseArgument()
        parameters = NameList(args.namelist)
        figureConfig = FigureConfig(args, parameters)
        make_directory("./rjci_plot")
        if figureConfig.output_directory != "./":
            make_directory("./rjci_plot/" + args.output_directory)
        if directory_exists("./rjci"):
            dataNum: int = 0
            while os.path.isfile(
                "./rjci/rjci_all_" + "{0:05d}".format(dataNum) + ".gz"
            ):
                filePath: str = "./rjci/rjci_all_" + "{0:05d}".format(dataNum) + ".gz"
                if figureConfig.lineFlag:
                    linePlot(parameters, figureConfig, filePath, dataNum)
                else:
                    colorScalePlot(parameters, figureConfig, filePath, dataNum)
                print("Num=" + "{0:05d}".format(dataNum), end="\r")
                dataNum += 1
        else:
            _ = sys.stderr.write("rjci ディレクトリが存在しません.")
    except RuntimeError as error:
        _ = sys.stderr.write(f"Error: {error}\n")
        sys.exit(1)
    return


if __name__ == "__main__":
    main()
