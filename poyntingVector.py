#!/usr/bin/python3

import numpy as np

# from scipy.constants import *  # 物理定数の利用
import argparse
from argparse import RawTextHelpFormatter  # オプションの改行用
from scipy import constants as sc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Color bar のサイズ調整用
from matplotlib.ticker import MultipleLocator
import f90nml
import os
import sys
import subprocess
import csv
import matplotlib

matplotlib.use("Agg")
from matplotlib.ticker import ScalarFormatter

help_desc_msg = """【概要】
ipicls2d によって計算された電磁場(Ex, Ey, Bz)を 2 次元カラーマップで empi_plot 内に保存し，そのアニメーションも作成する．
x 方向のポインティングベクトルを可視化する. """

parser = argparse.ArgumentParser(
    prog="empi.py",
    usage="empi.py [-h] [-s --snap] [-l --line] [-i --ex, -j --ey, -k --bz] \
        [-f --vmin][-t --vmax] [-x --xmin][-X --xmax] [-y --ymin] [-Y --ymax] \
            [-n --namelist] namelist_file",
    description=help_desc_msg,
    epilog="end",
    add_help=True,
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument("-n", "--namelist", type=str, required=True)
parser.add_argument(
    "-f",
    "--vmin",
    type=float,
    help="カラーバーの最小値・ラインプロットの最小値",
    action="store",
)

parser.add_argument(
    "-t",
    "--vmax",
    type=float,
    help="カラーバーの最大値・ラインプロットの最大値",
    action="store",
)

parser.add_argument(
    "-x", "--xmin", help="プロットする範囲(x軸の最小値)", action="store"
)

parser.add_argument(
    "-X", "--xmax", help="プロットする範囲(x軸の最大値)", action="store"
)

parser.add_argument(
    "-y", "--ymin", help="プロットする範囲(y軸の最小値)", action="store"
)

parser.add_argument(
    "-Y", "--ymax", help="プロットする範囲(y軸の最大値)", action="store"
)

parser.add_argument(
    "-s", "--snap", type=int, help="スナップショット時刻(整数を与える)", action="store"
)

# parser.add_argument('-E', '--Ey', help='レーザー電場をプロットする．', action='store_true')
# 引数を解析する(これは，お約束の記述)
args = parser.parse_args()

# //////////////////////////////// end of parse ///////////////////////////////
# ///// ////////////////////////// 物理定数の定義 ////////////////////////////////
pi = 3.14e0  # 円周率
epsilon_0 = 8.854e-12  # 真空の誘電率
e_charge = 1.6e-19  # 電子の素電荷   [C]
m_e = 9.1e-31  # 電子質量      [kg]
speed_of_light = 3.0e8  # 光速         [m/sec]
mu_0 = sc.mu_0
# /////////////////////////////////////////////////////////////////////////////

# namelist の解析 (using f90nml) namelist file (arg.namelist) を読み込んで、params に代入
params = f90nml.read(args.namelist)
print("読み込んだ namelist は", args.namelist, "です．")

# ////////////////////// namelist からの読み込み /////////////////////////////////
Nx = params["geom"]["Nx"]  # x 方向の全グリッド数
Ny = params["geom"]["Ny"]  # y 方向の全グリッド数
NV = params["geom"]["NV"]  # 左側の真空領域のグリッド数
NM = params["geom"]["NM"]  # プラズマ領域のグリッド数
boost_opt = params["geom"]["boost_opt"]  # boostオプション(論理)
nst_boost = params["geom"]["nst_boost"]  # simulation box がレーザーの系に乗る時刻
csim = params["geom"]["c"]  # ipicls2d での光速
system_lx = params["geom"]["system_lx"]  # x 方向のシステム長(delta_x x Nx)
system_ly = params["geom"]["system_ly"]  # y 方向のシステム長(delta_y x Ny)
wavelength = params["wave"]["wavelength"]  # 波長 [micron]
# diag
ndav = params["diag"]["ndav"]  # レーザーの1周期のタイムステップ
Nx_d = params["diag"]["Nx_d"]  # ファイルに出力するときの間隔
Ny_d = params["diag"]["Ny_d"]  # ファイルに出力するときの間隔
int_snap = params["diag"]["int_snap"]  # 出力ファイルの時間間隔
# wave
ow = params["wave"]["ow"]  # レーザーの角周波数
Exext = params["wave"]["Exext"]  # 初期印加電場(x方向)
Ezext = params["wave"]["Ezext"]  # 初期印加電場(z方向)
# ionize
ionize_opt = params["ionize"]["ionize_opt"]
zin0 = params["ionize"]["zin0"]
field_ionize_opt = params["ionize"]["field_ionize_opt"]
col_ionize_opt = params["ionize"]["col_ionize_opt"]
impact_ionize_opt = params["ionize"]["impact_ionize_opt"]
# ///////////////////// namelist からの読み込み終了 ///////////////////////////////

grid_x = int(Nx // Nx_d) + 1  # x方向のグリッド数
grid_y = int(Ny // Ny_d) + 1  # y方向のグリッド数
omega_L = (2 * pi * speed_of_light) / (wavelength * 1.0e-6)  # レーザー角周波数
n_cr = (m_e * epsilon_0 * omega_L**2) / e_charge**2 * 1e-6  # 臨界密度
n_pic = n_cr / ow**2
T = 2 * pi / ow  # レーザーの周期 (2π/ow)
d_t = T / ndav  # delta_time
# El = m_e * omega_L * c/e                            # z偏光のときElで計算
# /////////////////////////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////////////////////////
# Bx(1行目): particle = 0，By(2行目):  particle = 1, Ez(3行目): particle = 2
# initial : 初期印加電磁場(empiの場合はEzextのみ．Bx, By には初期磁場は設定できない)
# デフォルトは，Ez のマップを作成する(-i/-j/-kが指定されなかったときは，-k にする)
particle = 0
initial = Exext  # 初期印加電場
title = "Sx"
mag = 1  # mag は，規格化値からMKS単位系に変換するための係数となる予定
# /////////////////////////////////////////////////////////////////////////////////////////////////


# ////////////////////// カラーマップを作成する関数 ///////////////////////////////
def empi2D(datafile, DataNum, new_dir_path):  # カラーマップを作成する関数
    Sx_Output = np.zeros(
        [grid_y, grid_x]
    )  # 縦の長さが grid_y, 横の長さが grid_x のすべての要素が 0 の配列をつくる
    Sy_Output = np.zeros([grid_y, grid_x])
    Data = np.loadtxt(
        datafile
    )  # データを読み込む．loadtxt では 対象ファイルが gz の場合はまず解凍する．
    for i in range(grid_y):  # データを配列に格納
        for j in range(grid_x):
            Sx_Output[i, j] = (
                Data[i * grid_x + j, 1] * mag * Data[i * grid_x + j, 2] * mag
            ) / mu_0  # particle =0, 1, 2 -> Bx, By, Ez
    for i in range(grid_y):
        for j in range(grid_x):
            Sy_Output[i, j] = (
                -Data[i * grid_x + j, 0] * mag * Data[i * grid_x + j, 2] * mag
            ) / mu_0

    # グラフを描いて保存
    plt.rcParams["font.size"] = 20  # 字の大きさ font size
    f, ax = plt.subplots(figsize=(10, 6))  # 図を生成
    # 軸の設定
    if boost_opt is False:  # ブーストオプションを用いていない場合
        X = (
            np.linspace(
                -Sx_Output.shape[1] / 2, Sx_Output.shape[1] / 2, Sx_Output.shape[1] + 1
            )
            * (wavelength / ndav)
            * Nx_d
        )  # 中央を 0 にする
        Y = (
            np.linspace(
                Sx_Output.shape[0] / 2, -Sx_Output.shape[0] / 2, Sx_Output.shape[0] + 1
            )
            * (wavelength / ndav)
            * Ny_d
        )  # 中央を 0 にする
    else:
        if DataNum * int_snap <= nst_boost:  # ブーストオプションを用いている場合
            X = (
                np.linspace(-NV, Sx_Output.shape[1] - NV, Sx_Output.shape[1])
                * (wavelength / ndav)
                * Nx_d
            )  # ガス領域の始まりを 0 にする
        elif (
            nst_boost > (DataNum - 1) * int_snap and nst_boost < DataNum * int_snap
        ):  # 出力ファイル時間の間でブーストが始まるとき
            x_boost = (abs((DataNum * int_snap) - nst_boost)) * int_snap * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Sx_Output.shape[1] + x_boost - NV, Sx_Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )
        else:  # ブースト中
            x_boost = ((DataNum * int_snap) - nst_boost) * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Sx_Output.shape[1] + x_boost - NV, Sx_Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )
        Y = (
            np.linspace(
                -Sx_Output.shape[0] / 2, Sx_Output.shape[0] / 2, Sx_Output.shape[0]
            )
            * (wavelength / ndav)
            * Ny_d
        )

    if args.vmax:
        max_value = float(
            args.vmax
        )  # args.vmax は文字列なので，float で浮動小数に変換する．color bar の最大値
        min_value = float(args.vmin)  # color bar の最小値
        Im = ax.pcolorfast(
            X, Y, Sx_Output + initial, vmin=min_value, vmax=max_value, cmap="rainbow"
        )  # color map(contours:等高線図)  生成
    else:
        Im = ax.pcolorfast(
            X, Y, Sx_Output + initial, cmap="rainbow"
        )  # color map(contours:等高線図)  生成
    # pclorfast で，高さ方向の値で vmin=, vmax= で値を与えると，
    # それらを 最小値，最大値として高さ方向(色の濃淡)の図を作成する．

    # グラフの装飾
    ax.set_xlabel("x [$\mathrm{\mu m}$]")
    ax.set_ylabel("y [$\mathrm{\mu m}$]")
    ax.set_title("file=" + format(DataNum) + "     ", size="small", loc="right")
    real_time = DataNum * int_snap * wavelength * 1.0e-6 / 3.0e8 / ndav * 1.0e15
    time_step = int(DataNum * ndav)
    ax.set_title(
        f"{real_time:8.02f}" + " fsec=" + f"{time_step}" + " time step",
        size="small",
        loc="left",
    )
    ax.set_title(title, size="medium", loc="center")

    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
    ax.tick_params(which="both", right="on", top="on")
    ax.tick_params(which="major", direction="in", width=2, length=10, colors="k")
    ax.tick_params(which="minor", direction="in", length=5, colors="k")

    if args.xmin and args.xmax:  # x軸のプロット範囲の指定
        xmin_value = float(args.xmin)
        xmax_value = float(args.xmax)
        ax.set_xlim(xmin_value, xmax_value)  # xmin から xmax の範囲をプロット
    else:
        ax.set_xlim()
        ax.xaxis.set_major_locator(MultipleLocator(40))

    if args.ymin and args.ymax:  # y軸のプロット範囲の設定
        ymin_value = float(args.ymin)
        ymax_value = float(args.ymax)
        ax.set_ylim(ymin_value, ymax_value)  # ymin から ymax の範囲をプロット
    else:
        ax.set_ylim()
        ax.yaxis.set_major_locator(MultipleLocator(40))

    # Color bar
    Divider = make_axes_locatable(ax)
    Cax = Divider.append_axes("right", "5%", pad="2%")
    cbar = f.colorbar(Im, cax=Cax)
    cbar.set_label("Sx [$W/m^2$]", size=24)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    f.canvas.draw()
    step = 100
    meshX, meshY = np.meshgrid(X[:-1], Y[:-1])
    magnitude = np.sqrt(Sx_Output**2 + Sy_Output**2)
    magnitude[magnitude == 0] = 1e-30
    Sx_norm = Sx_Output / magnitude
    Sy_norm = Sy_Output / magnitude
    ax.quiver(
        meshX[::step, ::step],
        meshY[::step, ::step],
        Sx_norm[::step, ::step],
        Sy_norm[::step, ::step],
        color="white",
        scale=25,
    )
    plt.savefig(
        new_dir_path
        + "/"
        + title
        + "_poytingVector_"
        + "{0:05d}".format(DataNum)
        + ".png"
    )  # empi_plot に png で保存
    plt.close("all")
    del cbar, f, ax, Im, Data, Sx_Output, Sy_Output


# ////////////////////// カラーマップを作成する関数(終了) ///////////////////////////


# main function   /////////////////////////////////////////////////////////////
def main():
    FilePath = "./"  # namelist のファイルパス
    DataNum = 0  # 出力ファイルカウンターの初期化
    # path_namelist = (
    #     FilePath + args.namelist
    # )  # args.namelist は，コマンドラインで与えた namelist file 名
    print("与えられた namelist からパラメーターを読み込みました．\n")

    if args.snap:
        FilePath = "./"
        snap_value = int(args.snap)  # color bar の最小値
        snap_file = FilePath + "empi/empi_all_" + "{0:05d}".format(snap_value) + ".gz"
        print("指定時刻", snap_value)
        print("指定ファイル", snap_file)
        new_dir_path = FilePath + "empi_plot"  # 保存先のファイルパス
        os.makedirs(
            new_dir_path, exist_ok=True
        )  # 保存先のディレクトリ empi_plot を作成
        empi2D(
            snap_file, snap_value, new_dir_path
        )  # カラーマップ作成関数 empi2D を実行
        cmd6 = (
            "open ./empi_plot/"
            + title
            + "poytingVector"
            + "{0:05d}".format(snap_value)
            + ".png"
        )
        subprocess.call(cmd6.split())  # 時間指定したマップを表示

    # empi ファイルの読み込み
    # empi dir の存在の有無も判定したほうがいい？  os.path.isdir()
    if (
        os.path.isfile(FilePath + "empi/empi_all_" + "{0:05d}".format(DataNum) + ".gz")
        is True
    ):
        # os.path.isfile フィアルの存在確認 True => 存在している
        while os.path.isfile(
            FilePath + "empi/empi_all_" + "{0:05d}".format(DataNum) + ".gz"
        ):  # このwhile で各データをプロット．
            # while で，*.gz な全部のファイルに対して実行する
            datafile = (
                FilePath + "empi/empi_all_" + "{0:05d}".format(DataNum) + ".gz"
            )  # データファイルのパス
            # datafile2 = (
            #     FilePath
            #     + 'empi/empi_all_'
            #     + '{0:05d}'.format(DataNum)
            #     + '.gz'
            # )  # データファイルのパス
            new_dir_path = FilePath + "empi_plot"  # 保存先のファイルパス
            os.makedirs(
                new_dir_path, exist_ok=True
            )  # 保存先のディレクトリ empi_plot を作成
            print("Num=" + "{0:05d}".format(DataNum))  # プロット中のファイル時間を出力
            empi2D(
                datafile, DataNum, new_dir_path
            )  # カラーマップ作成関数 empi2D を実行
            DataNum += 1  # ファイル時間をカウントアップ

    cmd1 = (
        "ffmpeg -loglevel quiet -y -r 20 -i ./empi_plot/"
        + title
        + "_poytingVector_%5d.png -pix_fmt yuv420p ./empi_plot/"
        + title
        + "_animation.mp4"
    )
    cmd2 = "open ./empi_plot/" + title + "_animation.mp4"
    subprocess.call(cmd1.split())  # ffmpeg でanimation gif を生成
    subprocess.call(cmd2.split())  # animation gif をopen
    print(
        "アニメーションを empi_plot 以下に"
        + title
        + "_animation.mp4 として保存しました．"
    )
    # 入力したコマンドをファイルにする．
    f = open("./empi_plot/command_line.txt", "w")
    f.write(" ".join(sys.argv))  # join でlist -> str に変換
    f.close()
    # else:
    #     print('処理するデータファイルが empi ディレクトリにありません')


if __name__ == "__main__":
    main()
