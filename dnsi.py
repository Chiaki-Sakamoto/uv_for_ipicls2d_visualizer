#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ////////////////////////////////////////////////////////////////////////////////
#  Original code is  coded by Gunji
#
#  ipicls2d による計算結果で dnsi ディレクトリに保存される結果を2次元プロットするプログラム
#  プロットした結果は dnsi_plot に保存される．
#  dnsi file はレーザー1周期で平均した密度の値を
#  1 列目 イオン密度，2 列目 電子密度
#  で出力している．
#
# 使い方[test.py]
#   dnsi.py を PATH の通ったところに保存し
#    > chmod +x dnsi.py
#   で実行許可を与える．
#
#  計算のときに用いた namelist file を引数として与え，設定された値を読み込んでいる．
#     > dnsi.py [-l, -f -t] -n namelist_file
#
#   > dnsi.py -h または dnsi.py --help
#
#  で簡単な使い方の help が表示される．
#  動画ファイルを作成するには -p オプションで .png で保存する必要がある
#
#  version 0.1
#  version 0.2 (20230508)
#     ・2次元表示のデータの最小値と最大値を -f(--vmin), -t(--vmax) で設定できるようにした．
#     ・ffmpeg で作成するファイルを mp4 に変更した(パワポに貼れるように)
#  version 0.3 (20230524)
#     ・-i/--ion option を設定した．使わないかもしれないがイオン密度をプロットできるようにした．
#  version 0.4 (20230823)
#     ・-E option を設定した. レーザー波形を白く重ねてプロットできるようにした.
#     ・-x -y option を設定した. x軸とy軸のプロット範囲を設定できるようにした.
#  version 0.5 (20231124)
#     ・-E option のエラーを修正した
#     ・-x -X -y -Y option を設定した. x軸とy軸のプロット範囲(始点と終点)を設定できるようにした.
#  version 0.6 (20240522)
#     ・ -p/--png option を廃止し，図を png のみとした
# ///////////////////////////////////////////////////////////////

import numpy as np
import argparse
from argparse import RawTextHelpFormatter  # オプションの改行用
import f90nml
import os
import sys
import subprocess
import csv
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.constants import e
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Color bar のサイズ調整用
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MultipleLocator

# ///////////////// 外部コマンドのチェック //////////////////////////
# コマンドチェック see ~/bin/comand_check.py
# ret = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
# ret = subprocess.run(["which", "ffmpeg"], stderr=subprocess.STDOUT)
# if ret.returncode==1:
#     print("ffmpeg はインストールされていません．dnsi.py の実行には ffmpeg が必要です．")
#     exit()
# ///////////////// parse /////////////////////////////////////
help_desc_msg = """【概要】
ipicls2d によるプラズマ密度(デフォルトは電子密度)の結果 dnsi を2次元カラーマップで png 形式
で dnsi_plot 内に保存し，動画や与えたコマンドラインも保存する．
-i/--ion option の指定でイオン密度を選択する
-f/--vmin, -t/--vmax の指定でカラーバーの最大最小を指定
-x/--xmin, -X/--xmax, -y/--ymin, -Y/--ymax の指定でx軸とy軸のプロット範囲を指定
"""

parser = argparse.ArgumentParser(
    prog="dnsi.py",
    usage="dnsi.py [-h] [-l --line] [-i --ion] [-f --vmin] [-t --vmax] \
        [-x --xmin][-X --xmax] [-y --ymin] [-Y --ymax] \
            [-n --namelist] namelist_file",
    description=help_desc_msg,
    epilog="end",
    add_help=True,
    formatter_class=RawTextHelpFormatter,
)
_ = parser.add_argument(
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
_ = parser.add_argument(
    "-i",
    "--ion",
    help="イオン密度をプロット(デフォルトは電子密度)",
    action="store_true",
)
_ = parser.add_argument("-n", "--namelist", type=str, required=True)
_ = parser.add_argument(
    "-f", "--vmin", help="カラーバーの最小値(from). -t と同時に用いる", action="store"
)
_ = parser.add_argument(
    "-t", "--vmax", help="カラーバーの最大値(to). -f と同時に用いる", action="store"
)
_ = parser.add_argument(
    "-x", "--xmin", help="プロットする範囲(x方向の最小値)", action="store"
)
_ = parser.add_argument(
    "-X", "--xmax", help="プロットする範囲(x方向の最大値)", action="store"
)
_ = parser.add_argument(
    "-y", "--ymin", help="プロットする範囲(y方向の最小値)", action="store"
)
_ = parser.add_argument(
    "-Y", "--ymax", help="プロットする範囲(y方向の最大値)", action="store"
)
_ = parser.add_argument(
    "-E", "--Ey", help="レーザー電場をプロットする．", action="store_true"
)
_ = parser.add_argument(
    "-s", "--snap", type=int, help="スナップショット時刻(整数を与える)", action="store"
)

# 引数を解析する(これは，お約束の記述)
args = parser.parse_args()

# ///////////////// end of parse ///////////////////////////

# ///////// 物理定数の定義 ////////////////////////////////////
pi = 3.14e0  # 円周率
epsilon_0 = 8.854e-12  # 真空の誘電率
e_charge = 1.6e-19  # 電子の素電荷   [C]
m_e = 9.1e-31  # 電子質量      [kg]
speed_of_light = 3.0e8  # 光速         [m/sec]
# ///////////////////////////////////////////////////////////

# namelist の解析 (using f90nml) namelist file (arg.namelist) を読み込んで、params に代入
namelist_path: str = args.namelist
params: dict[str, dict[str, int | float | str]] = f90nml.read(namelist_path)

print("読み込んだ namelist は", namelist_path, "です．")

# ion option の処理：イオンなら particle = 0，電子なら particle = 1
if args.ion:
    particle = 0  # 0 は ion
    y_axis_title = "Ion Density"  # contorのz軸
    out_file_suffix = "i_density"  # png file の接頭語
    particle_index = "i_"  # 動画ファイルの接頭語
    title = "n_i"  # グラフの上部中央のタイトル
else:
    particle = 1  # 1 は electron
    y_axis_title = "Electron Density"
    out_file_suffix = "e_density"
    particle_index = "e_"
    title = "n_e"  # グラフの上部中央のタイトル

if particle == 0:
    print("イオン密度のマップを作成します．")
else:
    print("電子密度のマップを作成します．")

if args.line:
    print("x軸上の値のラインプロットのグラフを作成します")
# /////// namelist からの読み込み /////////////////////////////////////
Nx: int = params["geom"]["Nx"]  # x 方向の全グリッド数
Ny = params["geom"]["Ny"]  # y 方向の全グリッド数
NV = params["geom"]["NV"]  # 左側の真空領域のグリッド数
NM = params["geom"]["NM"]  # プラズマ領域のグリッド数
boost_opt = params["geom"]["boost_opt"]  # boostオプション(論理)
nst_boost = params["geom"]["nst_boost"]  # simulation box がレーザーの系に乗る時刻
csim = params["geom"]["c"]  # ipicls2d での光速
system_lx = params["geom"]["system_lx"]  # x 方向のシステム長(delta_x x Nx)
system_ly = params["geom"]["system_ly"]  # y 方向のシステム長(delta_y x Ny)
wavelength = params["wave"]["wavelength"]  # 波長 [micron]
ow = params["wave"]["ow"]  # レーザーの角周波数
ndav = params["diag"]["ndav"]  # レーザーの1周期のタイムステップ
Nx_d = params["diag"]["Nx_d"]  # ファイルに出力するときの間隔
Ny_d = params["diag"]["Ny_d"]  # ファイルに出力するときの間隔
int_snap = params["diag"]["int_snap"]  # 出力ファイルの時間間隔

# ionize
ionize_opt = params["ionize"]["ionize_opt"]
zin0 = params["ionize"]["zin0"]
field_ionize_opt = params["ionize"]["field_ionize_opt"]
col_ionize_opt = params["ionize"]["col_ionize_opt"]
impact_ionize_opt = params["ionize"]["impact_ionize_opt"]
# ////// namelist からの読み込み終了 ////////////////////////////

grid_x = int(Nx // Nx_d) + 1  # x方向のグリッド数
grid_y = int(Ny // Ny_d) + 1  # y方向のグリッド数
omega_L = (2 * pi * speed_of_light) / (wavelength * 1.0e-6)  # レーザー角周波数
n_cr = (m_e * epsilon_0 * omega_L**2) / e_charge**2 * 1e-6  # 臨界密度
n_pic = n_cr / ow**2
T = 2 * pi / ow  # レーザーの周期 (2π/ow)
d_t = T / ndav  # delta_time
El = m_e * omega_L * c / e  # z 偏光のとき El で計算


# //// function dnsi2D  ///////////////////////////////////////////////////////
#
#    マップ作成のための関数  :      dnsi2D
#
# /////////////////////////////////////////////////////////////////////////////
def dnsi2D(datafile, DataNum, new_dir_path):
    Output = np.zeros(
        [grid_y, grid_x]
    )  # 縦の長さが grid_y, 横の長さが grid_x のすべての要素が 0 の配列をつくる
    Data = np.loadtxt(
        datafile
    )  # データを読み込む．loadtxt では 対象ファイルが gz の場合はまず解凍する．
    #    print(Data)
    for i in range(grid_y):  # データを配列に格納
        for j in range(grid_x):
            Output[i, j] = (
                Data[i * grid_x + j, particle] * n_pic
            )  # particle =0 ion density
            # particle =1 electron density

    # グラフを描いて保存
    plt.rcParams["font.size"] = 20  # font size =20
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
            * Nx_d
        )

    if args.vmin and args.vmax:
        min_value = float(
            args.vmin
        )  # args.vmin は文字列なので，float で浮動小数に変換する．
        max_value = float(
            args.vmax
        )  # args.vmax は文字列なので，float で浮動小数に変換する．
        Im = ax.pcolorfast(
            X, Y, Output, vmin=min_value, vmax=max_value, cmap="rainbow"
        )  # color map(contours:等高線図)  生成
    else:
        Im = ax.pcolorfast(
            X, Y, Output, cmap="rainbow"
        )  # color map(contours:等高線図)  生成(カラーバーの指定なし)

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
        cbar.set_label(y_axis_title + " [$\mathrm{cm^{-3}}$]", size=24)
    else:
        cbar.set_label(y_axis_title + " [$\mathrm{cm^{-3}}$]", size=24)

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    plt.tight_layout()
    f.canvas.draw()

    plt.savefig(
        new_dir_path + out_file_suffix + "{0:05d}".format(DataNum) + ".png"
    )  # dnsi_plot に png で保存
    plt.close("all")
    del cbar, f, ax, Im, Data, Output


# ///////// function laser_E /////////////////////////////////////////////////
#
#           laser_E
#
# プラズマ密度とレーザー波形を重ねて描写する これは  Ey と同じなので，削除した(20250325)．
#
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////
# def laser_E(datafile, datafile2, DataNum, new_dir_path): #プラズマ密度とレーザー波形を重ねて描写
#     Output = np.zeros([grid_y, grid_x]) #縦の長さがgrid_y,横の長さがgrid_xの配列をつくる
#     Data = np.loadtxt(datafile) #データを読み込み
#     for i in range (grid_y): #データを配列に格納
#         for j in range (grid_x):
#             Output[i, j] = Data[i *grid_x + j, 1] * n_pic #n_pic をかけて元の値に戻す
#     Output2 = np.zeros([grid_y, grid_x]) #縦の長さがgrid_y,横の長さがgrid_xの配列をつくる
#     Data2 = np.loadtxt(datafile2) #データを読み込み
# for k in range(grid_y):  # データを配列に格納
#     for l in range(grid_x):
#         Output2[k, l] = Data2[k *grid_x + l, 1] * El / 1e12
#     outmax=np.amax(Output2)
#     outmin=np.amin(Output2)

#     #グラフを描いて保存
#     plt.rcParams['font.size'] = 15 #字の大きさ
#     f = plt.figure(figsize = (12,6),constrained_layout=True)
#     ax1 = f.subplots()#図を生成
#     ax2 = ax1.twinx()
#     #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

#     #軸の設定
# if boost_opt == False: #ブーストオプションを用いていない場合
#     X = np.linspace(
#         -Output.shape[1]/2, Output.shape[1]/2+1, Output.shape[1]
#     ) * wavelength / ndav #中央を 0 にする
#     Y = np.linspace(
#         Output.shape[0]/2, -Output.shape[0]/2+1, Output.shape[0]
#     ) * wavelength / ndav #中央を 0 にする
# else:
#     if DataNum * int_snap <= nst_boost:  #ブーストオプションを用いている場合
#         X = np.linspace(
#             -NV, Output.shape[1]-NV, Output.shape[1]
#         ) * wavelength / ndav #ガス領域の始まりを 0 にする
#     elif (
#         nst_boost > (DataNum-1) * int_snap and
#         nst_boost < DataNum * int_snap
#     ):  # 出力ファイル時間の間でブーストが始まるとき
#         x_boost =  (
#             abs((DataNum * int_snap) - nst_boost)
#         ) * int_snap * csim * d_t
#         X = np.linspace(
#             x_boost-NV, Output.shape[1]+x_boost-NV, Output.shape[1]
#         ) * wavelength / ndav
#     else:  # ブースト中
#         x_boost = ((DataNum * int_snap) - nst_boost) * csim * d_t
#         X = np.linspace(
#             x_boost - NV, Output.shape[1] + x_boost - NV, Output.shape[1]
#         ) * wavelength / ndav
#     Y = np.linspace(
#         -Output.shape[0]/2, Output.shape[0]/2, Output.shape[0]
#     ) * wavelength / ndav
#     y = grid_y // 2
# # Im = ax.pcolorfast(
# #     X, Y, Output, cmap = 'rainbow', norm=LogNorm()
# # )  # カラーマップの生成
# Im = ax1.pcolorfast(
#     X, Y, Output, cmap = 'rainbow'
# )  # pclorfast による contours:等高線図生成
# ax2.plot(
#     X, Output2[y,:], c='w',  label = '$E_x$', alpha=0.6
# )
#     #Im = ax.pcolormesh(X, Y, Output, cmap = 'rainbow', norm=LogNorm())
#     #cbar = f.colorbar(mappable0, orientation="vertical")
#     Y2_max = np.amax(Output2[y,:]) + np.amax(Output2[y,:]) * 0.2
#     Y2_min = np.amin(Output2[y,:]) + np.amin(Output2[y,:]) * 0.2

# #グラフの装飾
# ax1.set_xlabel('x [$\mathrm{\mu m}$]')
# ax1.set_ylabel('y [$\mathrm{\mu m}$]')
# ax2.set_ylabel('Electric field $E_x$ [GV/m]')
# #ax2.set_ylim(-6, 6)
# #ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
# #ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(100))
# ax1.tick_params(which='both',right='on',top='on')
# ax1.tick_params(
#     which='major', direction='in', width=2, length=10, colors='k'
# )
# ax1.tick_params(
#     which='minor', direction='in', length=5, colors='k'
# )
#     #ax2.spines["right"].set_position(("axes", 1.5))

#     if args.xmin and args.xmax: #x軸のプロット範囲の指定
#         xmin_value = float(args.xmin)
#         xmax_value = float(args.xmax)
#         ax1.set_xlim(xmin_value,xmax_value)  # xmin から xmax の範囲をプロット
#         ax2.set_xlim(xmin_value,xmax_value)  # xmin から xmax の範囲をプロット
#     else:
#         ax1.set_xlim()
#         ax2.set_xlim()

#     if args.ymin and args.ymax: #y軸のプロット範囲の指定
#         ymin_value = float(args.ymin)
#         ymax_value = float(args.ymax)
#         ax1.set_ylim(ymin_value,ymax_value)  # ymin から ymax の範囲をプロット
#         ax2.set_ylim(-6, 6)
#     else:
#         ax1.set_ylim()
#         ax1.set_ylim(-6, 6)

#     #Color bar
#     Divider = make_axes_locatable(ax1)
#     #Cax = Divider.append_axes()
#     cbar = plt.colorbar(Im)
#     cbar.set_label(y_axis_title+' [$\mathrm{cm^{-3}}$]',size=15)
#     cbar.ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:.2f}"))
#     cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#     cbar.ax.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
#     f.canvas.draw()
# #    if args.png:
#     plt.savefig(
#         new_dir_path + '/dnsi_Ey'+ '{0:05d}'.format(DataNum) +'.png'
#     )  # dnsi_plot に png で保存
#     cmd5 = ('ffmpeg -loglevel quiet -y -r 20 -i'
#             + './dnsi_plot/dnsi_Ey%5d.png -pix_fmt yuv420p'
#             + './dnsi_plot/output_Ey.mp4'
#         )
# # -y : 強制的に上書きする
#     cmd6 = 'open ./dnsi_plot/output_Ey.mp4'
#     subprocess.call(cmd5.split())  # ffmpeg でanimationを生成
#     print("x軸での変化を dnsi_plot 以下に output_Ey.mp4 として保存しました．")
# #   else:
#         # plt.savefig(
#         #     new_dir_path + '/dnsi_Ey'+ '{0:05d}'.format(DataNum) + '.pdf'
#         # )  # dnsi_plot に pdf で保存
#     plt.close()


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# x 軸上での電子(イオン)密度の散布図を作成する関数
#
#  dnsi_line
#
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def dnsi_line(datafile, DataNum, new_dir_path):
    Output = np.zeros(
        [grid_y, grid_x]
    )  # 縦の長さがgrid_y,横の長さがgrid_xの配列をつくる
    Data = np.loadtxt(datafile)  # データの読み込み
    for i in range(grid_y):
        for j in range(grid_x):
            Output[i, j] = (
                Data[i * grid_x + j, particle] * n_pic
            )  # ここ 2 は間違いではないか? particle に修正した

    # outmax は利用しないのでコメントアウトします
    # outmax = np.amax(Output)  # 最大値
    # outmin = np.amin(Output)  # 最小値
    # グラフを描いて保存
    plt.rcParams["font.size"] = 20
    f, ax = plt.subplots(figsize=(10, 6))
    # Making axis data
    # Making axis data
    if boost_opt is False:  # ブーストオプションを用いていない場合
        X = (
            np.linspace(-Output.shape[1] / 2, Output.shape[1] / 2, Output.shape[1])
            * (wavelength / ndav)
            * Nx_d
        )  # 中央を 0 にする
    else:
        if DataNum * int_snap <= nst_boost:  # ブーストオプションを用いている場合
            X = (
                np.linspace(-NV, Output.shape[1] - NV, Output.shape[1])
                * wavelength
                / ndav
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
    ax.plot(X, Output[y, :], color="red")  # y 軸の中央をプロット

    # csv dir の作成と  csv ファイルの保存 (dnsi_line では csv file を作成している for gnuplot)
    cmd7 = "mkdir -p csv"
    subprocess.call(cmd7.split())  # 時間指定したマップを表示
    with open(
        "csv/" + out_file_suffix + "{0:05d}".format(DataNum) + ".csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(zip(X, Output[y, :]))  # 各行に x と y のペアを書き込む

    # グラフの装飾
    ax.set_xlabel("x [$\mathrm{\mu m}$]")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # ax.set_ylabel('Electron density [x $ 10^{18} \mathrm{cm^{-3}}$]')
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    plt.minorticks_on()
    if args.xmin and args.xmax:
        xmin_value = float(args.xmin)
        xmax_value = float(args.xmax)
        ax.set_xlim(xmin_value, xmax_value)
    else:
        ax.set_xlim(np.amin(X), np.amax(X))
        ax.xaxis.set_major_locator(MultipleLocator(40))
    if args.ymax:
        Y_ma = float(args.ymax)
        Y_mi = float(args.ymin)
    else:
        Y_ma = np.amax(Output[y, :]) + np.amax(Output[y, :]) * 0.1
        Y_mi = np.amin(Output[y, :]) - np.amin(Output[y, :]) * 0.1
    ax.set_ylim(
        Y_mi, Y_ma
    )  # 上下±10%    -l, --line を指定した時 warning が出る．             ここを3回通っている
    ax.set_ylabel(y_axis_title + " [$\mathrm{cm^{-3}}$]", size=24)
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    plt.tight_layout()
    ax.tick_params(which="both", right="on", top="on")
    ax.tick_params(which="major", direction="in", width=2, length=10, colors="k")
    ax.tick_params(which="minor", direction="in", length=5, colors="k")
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    f.canvas.draw()
    plt.savefig(
        new_dir_path
        + "/"
        + out_file_suffix
        + "_line_"
        + "{0:05d}".format(DataNum)
        + ".png",
        transparent=True,
        bbox_inches="tight",
    )  # dnsi_plot に 散布図を n_e_iline_Num.png で保存
    plt.close()


# main function   //////////////////////////////////////////
def main():
    FilePath = "./"  # namelist のファイルパス=current dir
    new_dir_path = FilePath + "dnsi_plot/"  # 保存先のファイルパス
    DataNum = 0  # 出力ファイルカウンターの初期化
    # path_namelist は利用しないのでコメントアウト
    # path_namelist = (
    #     FilePath + args.namelist
    # )  # args.namelist は，コマンドラインで与えた namelist file 名
    print("与えられた namelist からパラメーターを読み込みました．\n")

    if args.snap:
        FilePath = "./"
        snap_value = int(args.snap)  # color bar の最小値
        snap_file = (
            FilePath + "dnsi/dnsi_all_" + "{0:05d}".format(snap_value) + ".gz"
        )  # snap の対象ファイル
        print("指定時刻", snap_value)
        print("指定ファイル", snap_file)
        new_dir_path = FilePath + "dnsi_plot/"  # 保存先のファイルパス
        os.makedirs(
            new_dir_path, exist_ok=True
        )  # 保存先のディレクトリ dnsi_plot を作成
        if args.line:  # -l or -line のとき，
            dnsi_line(snap_file, snap_value, new_dir_path)  # ラインプロットの関数の実行
            cmd5 = (
                "open ./dnsi_plot/"
                + out_file_suffix
                + "_line_"
                + "{0:05d}".format(snap_value)
                + ".png"
            )
            subprocess.call(cmd5.split())  # 時間指定したlineplot を表示
        else:
            dnsi2D(
                snap_file, snap_value, new_dir_path
            )  # カラーマップ作成関数 dnsi2D を実行
            cmd6 = (
                "open ./dnsi_plot/"
                + out_file_suffix
                + "{0:05d}".format(snap_value)
                + ".png"
            )
            subprocess.call(cmd6.split())  # 時間指定したマップを表示
        quit()

    # dnsi ファイルの読み込み
    # dnsi dir の存在の有無も判定したほうがいい？  os.path.isdir()
    if (
        os.path.isfile(FilePath + "dnsi/dnsi_all_" + "{0:05d}".format(DataNum) + ".gz")
        is True
    ):
        # os.path.isfile フィアルの存在確認 True => 存在している
        while os.path.isfile(
            FilePath + "dnsi/dnsi_all_" + "{0:05d}".format(DataNum) + ".gz"
        ):
            # while で，*.gz な全部のファイルに対して実行する
            datafile = (
                FilePath + "dnsi/dnsi_all_" + "{0:05d}".format(DataNum) + ".gz"
            )  # データファイルのパス
            # datafile2 は利用していないのでコメントアウト
            # datafile2 = (
            #     FilePath
            #     + "empi/empi_all_"
            #     + "{0:05d}".format(DataNum)
            #     + ".gz"
            # )  # データファイルのパス for E option
            new_dir_path = FilePath + "dnsi_plot/"  # 保存先のファイルパス
            os.makedirs(
                new_dir_path, exist_ok=True
            )  # 保存先のディレクトリ dnsi_plot を作成
            print("Num=" + "{0:05d}".format(DataNum))  # プロット中のファイル時間を出力
            if args.line:  # -l or -line のとき，
                dnsi_line(datafile, DataNum, new_dir_path)  # ラインプロットの関数の実行
            # elif args.Ey:  # -E のとき
            # laser_E(
            #     datafile, datafile2, DataNum, new_dir_path
            # )  # レーザー波形もプロット
            else:  # -l or --line でないとき，
                dnsi2D(
                    datafile, DataNum, new_dir_path
                )  # カラーマップ作成関数 dnsi2D を実行
            DataNum += 1  # ファイル時間をカウントアップ

    if args.line:  # -l or -line のとき，
        cmd3 = (
            "ffmpeg -loglevel quiet -y -r 20 -i ./dnsi_plot/"
            + out_file_suffix
            + "_line_%5d.png -pix_fmt yuv420p ./dnsi_plot/"
            + out_file_suffix
            + "_lineplot.mp4"
        )
        cmd4 = "open ./dnsi_plot/" + out_file_suffix + "_lineplot.mp4"
        subprocess.call(cmd3.split())  # ffmpeg でanimationを生成
        subprocess.call(cmd4.split())  # animation をopen
        print(
            "ラインプロットのアニメーションを dnsi_plot 以下に"
            + out_file_suffix
            + "_lineplot.mp4 として保存しました．"
        )
        f = open("./dnsi_plot/command_line.txt", "w")
        f.write(" ".join(sys.argv))  # join でlist -> str に変換
        f.close()
    else:  # -l or --line でないとき(デフォルトのマップのアニメーション)
        cmd1 = (
            "ffmpeg -loglevel quiet -y -r 20 -i ./dnsi_plot/"
            + out_file_suffix
            + "%5d.png -pix_fmt yuv420p ./dnsi_plot/"
            + out_file_suffix
            + "_animation.mp4"
        )
        cmd2 = "open ./dnsi_plot/" + out_file_suffix + "_animation.mp4"
        subprocess.call(cmd1.split())  # ffmpeg でanimation gif を生成
        subprocess.call(cmd2.split())  # animation gif をopen
        print(
            "アニメーションを dnsi_plot 以下に"
            + out_file_suffix
            + "_animation.mp4 として保存しました．"
        )
        # 入力したコマンドをファイルにする．
        f = open("./dnsi_plot/command_line.txt", "w")
        f.write(" ".join(sys.argv))  # join でlist -> str に変換
        f.close()


if __name__ == "__main__":
    main()
