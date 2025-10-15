#!/usr/bin/python3
# -*- coding: utf-8 -*-
# /////////////////////////////////////////////////////////////////////////////
#                                                                             /
#   empi.py                                                                   /
#                                                                             /
# /////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////
#  このプログラムは卒業生の飯田紘一君が作成し，郡司貴大君と西田大紀君が改良した pic2d_dnssplot.py
#  である．この3名に感謝．
#
#  ipicls2d による計算結果で empi ディレクトリに保存される電磁場の結果を2次元プロットするプログラム
#  プロットした結果と動画およびコマンドラインで与えたオプションは empi_plot に保存される．
#  empi file は瞬時値の値であり，
#  1 列目 Bx，2 列目 By, 3列目 Ez
#  で出力している．
#
# 使い方[test.py]
#   empi.py を PATH の通ったところに保存し
#    > chmod +x empi.py
#   で実行許可を与える．
#
#  計算のときに用いた namelist file を引数として与え，設定された値を読み込んでいる．
#   > empi.py [-h] [-l --line] [-s --snap] [-i --ex] [-j --ey] [-b --bz] \
#        [-t --vmax] [-x --xmin][-X --xmax] [-y --ymin] [-Y --ymax] \
#            [-n --namelist] namelist_file
#
#   > empi.py -h または empi.py --help
#
#  で簡単な使い方の help が表示される．
#
#   emsi.py -> empi.py を作成した．基本的に emsi.py と同じ
#
#  todo:
#      -l /--line のときのx軸の範囲，y軸の範囲の設定ができるようにする．
#      グラフの値を規格化した値ではなく，SI単位系で出力できるようにする (V/m や Wb/m^2 のように)
# //////////////////////////////////////////////////////////////////////////////////

import numpy as np

# from scipy.constants import *  # 物理定数の利用
import argparse
from argparse import RawTextHelpFormatter  # オプションの改行用
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

# ////////////////////////// 外部コマンドのチェック ////////////////////////////////
# コマンドチェック see ~/bin/comand_check.py
# ret = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
# ret = subprocess.run(["which", "ffmpeg"], stderr=subprocess.STDOUT)
# if ret.returncode==1:
#     print("ffmpeg はインストールされていません．empi.py の実行には ffmpeg が必要です．")
#     exit()
# ////////////////////////////// parse ////////////////////////////////////////
help_desc_msg = """【概要】
ipicls2d によって計算された電磁場(Ex, Ey, Bz)を 2 次元カラーマップで empi_plot 内に保存し，そのアニメーションも作成する．
-i/--ex で Ex, -j/--ey で Ey，-b/--bz で Bz が得られる．
-x/--xmin, -X/--xmax, -y/--ymin, -Y/--ymax の指定でx軸とy軸のプロット範囲を指定することができる．
-f/--vmin, -t/--vmax でマップのカラーバーの範囲(最小値，最大値)，ラインプロットの場合は縦軸の最小値と最大値(f=from t=to)
ただし，-f/--vmin と -t/--vmax は同時に両方指定しなければならない．
-s/--snap 整数により任意の時刻の図(デフォルトはカラーマップ)も得ることができる．
-l/--line でx軸上(y=0)でのラインプロット．csv dir にそのときのデータファイルを出力される．
【注意】負の数はダブルコーテーションで囲み，かつマイナスの前に半角スペースを入れること．"""

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

parser.add_argument(
    "-l",
    "--line",
    help="各時刻における y=0 での物理量をラインプロットする．",
    action="store_true",
)
# parser.add_argument(
#     '-p',
#     '--png',
#     help='グラフを png で保存し，動画ファイルを作成する．',
#     action='store_true'
# )

parser.add_argument("-i", "--ex", help="Ex を出力", action="store_true")
parser.add_argument("-j", "--ey", help="Ey を出力", action="store_true")
parser.add_argument("-k", "--bz", help="Bz を出力", action="store_true")
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
title = "Ex"
mag = 1  # mag は，規格化値からMKS単位系に変換するための係数となる予定
# /////////////////////////////////////////////////////////////////////////////////////////////////

if args.ex:  # 1 行目のデータ
    particle = 0
    title = "Ex"
    initial = Exext
    mag = 1
elif args.ey:  # 2 行目のデータ
    particle = 1
    title = "Ey"
    initial = 0
    mag = 1
elif args.bz:  # 3 行目のデータ
    particle = 2
    title = "Bz"
    initial = 0
    mag = 1
# m_e * speed_of_light * speed_of_light  * 2
# * pi / e_charge /(wavelength * 1e-6)
if particle == 0:
    print("Ex のマップを作成します．")
if particle == 1:
    print("Ey のマップを作成します．")
if particle == 2:
    print("Bz のマップを作成します．")

if args.line:
    print("x軸上の値のラインプロットのグラフを作成します")

# ////////////////////// カラーマップを作成する関数 ///////////////////////////////


def empi2D(datafile, DataNum, new_dir_path):  # カラーマップを作成する関数
    Output = np.zeros(
        [grid_y, grid_x]
    )  # 縦の長さが grid_y, 横の長さが grid_x のすべての要素が 0 の配列をつくる
    Data = np.loadtxt(
        datafile
    )  # データを読み込む．loadtxt では 対象ファイルが gz の場合はまず解凍する．
    for i in range(grid_y):  # データを配列に格納
        for j in range(grid_x):
            Output[i, j] = (
                Data[i * grid_x + j, particle] * mag
            )  # particle =0, 1, 2 -> Bx, By, Ez

    # グラフを描いて保存
    plt.rcParams["font.size"] = 20  # 字の大きさ font size
    f, ax = plt.subplots(figsize=(10, 6))  # 図を生成
    # 軸の設定
    if boost_opt is False:  # ブーストオプションを用いていない場合
        X = (
            np.linspace(-Output.shape[1] / 2, Output.shape[1] / 2, Output.shape[1] + 1)
            * (wavelength / ndav)
            * Nx_d
        )  # 中央を 0 にする
        Y = (
            np.linspace(Output.shape[0] / 2, -Output.shape[0] / 2, Output.shape[0] + 1)
            * (wavelength / ndav)
            * Ny_d
        )  # 中央を 0 にする
    else:
        if DataNum * int_snap <= nst_boost:  # ブーストオプションを用いている場合
            X = (
                np.linspace(-NV, Output.shape[1] - NV, Output.shape[1])
                * (wavelength / ndav)
                * Nx_d
            )  # ガス領域の始まりを 0 にする
        elif (
            nst_boost > (DataNum - 1) * int_snap and nst_boost < DataNum * int_snap
        ):  # 出力ファイル時間の間でブーストが始まるとき
            x_boost = (abs((DataNum * int_snap) - nst_boost)) * int_snap * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Output.shape[1] + x_boost - NV, Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )
        else:  # ブースト中
            x_boost = ((DataNum * int_snap) - nst_boost) * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Output.shape[1] + x_boost - NV, Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )
        Y = (
            np.linspace(-Output.shape[0] / 2, Output.shape[0] / 2, Output.shape[0])
            * (wavelength / ndav)
            * Ny_d
        )

    if args.vmax:
        max_value = float(
            args.vmax
        )  # args.vmax は文字列なので，float で浮動小数に変換する．color bar の最大値
        min_value = float(args.vmin)  # color bar の最小値
        Im = ax.pcolorfast(
            X, Y, Output + initial, vmin=min_value, vmax=max_value, cmap="rainbow"
        )  # color map(contours:等高線図)  生成
    else:
        Im = ax.pcolorfast(
            X, Y, Output + initial, cmap="rainbow"
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

    if particle == 0:
        cbar.set_label("Ex [norm.]", size=24)
    if particle == 1:
        cbar.set_label("Ey [norm.]", size=24)
    if particle == 2:
        cbar.set_label("Bz [norm.]", size=24)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    f.canvas.draw()
    plt.savefig(
        new_dir_path + "/" + title + "_empi_" + "{0:05d}".format(DataNum) + ".png"
    )  # empi_plot に png で保存
    plt.close("all")
    del cbar, f, ax, Im, Data, Output


# ////////////////////// カラーマップを作成する関数(終了) ///////////////////////////


# ////////////////////// ラインプロットを作成する関数 ///////////////////////////////
#
#  empi_line
#
# ///////////////////////////////////////////////////////////////////////////////////////////
def empi_line(datafile, DataNum, new_dir_path):
    Output = np.zeros(
        [grid_y, grid_x]
    )  # 縦の長さがgrid_y,横の長さがgrid_xの配列をつくる

    Data = np.loadtxt(datafile)  # データの読み込み
    for i in range(grid_y):
        for j in range(grid_x):
            Output[i, j] = (
                Data[i * grid_x + j, particle] + initial
            )  # ここ 2 は間違いではないか? particle に修正した
    # outmax は利用されていないのでコメントアウトします
    # if args.vmax:
    #     outmax = float(args.vmax)  # ラインプロットの場合は，縦軸の最大値
    #     outmin = float(args.vmin)  # ラインプロットの場合は，縦軸の最小値
    # else:
    #     outmax = np.amax(Output)  # 最大値
    #     outmin = np.amin(Output)  # 最小値
    # グラフを描いて保存
    plt.rcParams["font.size"] = 20
    f, ax = plt.subplots(figsize=(10, 6))
    # Making axis data (linspace(start，stop，num)
    # start から stop まで num 個の要素を生成する )
    if boost_opt is False:  # ブーストオプションを用いていない場合
        X = (
            np.linspace(-Output.shape[1] // 2, Output.shape[1] // 2, Output.shape[1])
            * (wavelength / ndav)
            * Nx_d
        )  # 中央を 0 にする
        print(
            "linespace",
            np.linspace(-Output.shape[1] // 2, Output.shape[1] // 2, Output.shape[1]),
        )
        print("X の要素数 =", len(X))
        print("X の配列 =", X)
        dx = X[1] - X[0]
        print(f"dx = {dx} μm, wavelength = {wavelength}")
    else:
        if DataNum * int_snap <= nst_boost:  # ブーストオプションを用いている場合
            X = (
                np.linspace(-NV, Output.shape[1] - NV, Output.shape[1])
                * (wavelength / ndav)
                * Nx_d
            )
        elif nst_boost > (DataNum - 1) * int_snap and nst_boost < DataNum * int_snap:
            x_boost = (abs((DataNum * int_snap) - nst_boost)) * int_snap * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Output.shape[1] + x_boost - NV, Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )
        else:
            x_boost = ((DataNum * int_snap) - nst_boost) * csim * d_t
            X = (
                np.linspace(
                    x_boost - NV, Output.shape[1] + x_boost - NV, Output.shape[1]
                )
                * (wavelength / ndav)
                * Nx_d
            )

    y = grid_y // 2  # y 軸の中央
    ax.plot(X, Output[y, :], "o-")  # y 軸の中央をプロット
    # print(X)    # X でx軸の値を決めている．
    # print(Output[y,:])
    # print(Output.shape[1])   # x軸の要素の個数
    # 横軸を変えられないので，csv ファイルを出力するようにした．
    # csv dir の作成
    cmd7 = "mkdir -p csv"
    subprocess.call(cmd7.split())  # 時間指定したマップを表示
    with open(
        "csv/" + title + "_empi_" + "{0:05d}".format(DataNum) + ".csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(zip(X, Output[y, :]))  # 各行に x と y のペアを書き込む
    # print(
    #     "Data for lineplot is saved as  csv/empi_"
    #     + '{0:05d}'.format(DataNum)
    #     + ".csv"
    # )  # プロット中のファイル時間を出力

    # グラフの装飾
    if args.xmin and args.xmax:
        xmin_value = float(args.xmin)
        xmax_value = float(args.xmax)
        ax.set_xlim(xmin_value, xmax_value)
    else:
        ax.set_xlim(np.amin(X), np.amax(X))
    ax.set_xlabel("x [$\mathrm{\mu m}$]")
    ax.set_ylabel(ylabel="Ey [$\mathrm{norm .}$]")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.minorticks_on()
    if args.ymin and args.ymax:
        Y_mi = float(args.ymin)
        Y_ma = float(args.ymax)
    else:
        Y_ma = np.amax(Output[y, :]) + np.amax(Output[y, :]) * 0.1
        Y_mi = np.amin(Output[y, :]) - np.amin(Output[y, :]) * 0.1
    ax.set_ylim(Y_mi, Y_ma)  # 上下±10%
    ax.tick_params(which="both", right="on", top="on")
    ax.tick_params(which="major", direction="in", width=2, length=10, colors="k")
    ax.tick_params(which="minor", direction="in", length=5, colors="k")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    real_time = DataNum * int_snap * wavelength * 1.0e-6 / 3.0e8 / ndav * 1.0e15
    time_step = int(DataNum * ndav)
    ax.set_title(
        f"{real_time:8.02f}" + " fsec=" + f"{time_step}" + " time step",
        size="small",
        loc="left",
    )
    ax.set_title("file=" + format(DataNum) + "     ", size="small", loc="right")
    plt.tight_layout()
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    f.canvas.draw()
    #  グラフの保存
    plt.savefig(
        new_dir_path + "/" + title + "_empiline_" + "{0:05d}".format(DataNum) + ".png"
    )  # empi_plot に png で保存
    plt.close()


# ////////////////////// ラインプロットを作成する関数(終了) /////////////////////////


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
        if args.line:  # -l or -line のとき，
            empi_line(snap_file, snap_value, new_dir_path)  # ラインプロットの関数の実行
            cmd5 = (
                "open ./empi_plot/"
                + title
                + "_empiline_"
                + "{0:05d}".format(snap_value)
                + ".png"
            )
            subprocess.call(cmd5.split())  # 時間指定したlineplot を表示
        else:
            empi2D(
                snap_file, snap_value, new_dir_path
            )  # カラーマップ作成関数 empi2D を実行
            cmd6 = (
                "open ./empi_plot/"
                + title
                + "_empi_"
                + "{0:05d}".format(snap_value)
                + ".png"
            )
            subprocess.call(cmd6.split())  # 時間指定したマップを表示
        quit()

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
            if args.line:  # -l or -line のとき，
                empi_line(datafile, DataNum, new_dir_path)  # ラインプロットの関数の実行
            # elif args.Ey:  # -E のとき
            # laser_E(
            #     datafile,
            #     datafile2,
            #     DataNum,
            #     new_dir_path
            # )  # レーザー波形もプロット
            else:  # -l or --line でないとき，
                empi2D(
                    datafile, DataNum, new_dir_path
                )  # カラーマップ作成関数 empi2D を実行
            DataNum += 1  # ファイル時間をカウントアップ

    if args.line:  # -l or -line のとき，
        cmd3 = (
            "ffmpeg -loglevel quiet -y -r 20 -i ./empi_plot/"
            + title
            + "_empiline_%5d.png -pix_fmt yuv420p ./empi_plot/"
            + title
            + "_lineplot.mp4"
        )
        cmd4 = "open ./empi_plot/" + title + "_lineplot.mp4"
        subprocess.call(cmd3.split())  # ffmpeg でanimationを生成
        subprocess.call(cmd4.split())  # animation をopen
        print(
            "ラインプロットのアニメーションを empi_plot 以下に"
            + title
            + "_lineplot.mp4 として保存しました．"
        )
        f = open("./empi_plot/command_line.txt", "w")
        f.write(" ".join(sys.argv))  # join でlist -> str に変換
        f.close()
    else:  # -l or --line でないとき(デフォルトのマップのアニメーション)
        cmd1 = (
            "ffmpeg -loglevel quiet -y -r 20 -i ./empi_plot/"
            + title
            + "_empi_%5d.png -pix_fmt yuv420p ./empi_plot/"
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
