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
matplotlib.rcParams["font.family"] = "Cica"
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
title = "waveDirectional"
mag = 1  # mag は，規格化値からMKS単位系に変換するための係数となる予定
# /////////////////////////////////////////////////////////////////////////////////////////////////


# ////////////////////// カラーマップを作成する関数 ///////////////////////////////
def directionalPlot(datafile, DataNum, new_dir_path):
    # --- データ読み込み ---
    Data = np.loadtxt(datafile)

    # --- 出力配列の初期化 ---
    Sx_Output = np.zeros((grid_y, grid_x))
    Sy_Output = np.zeros((grid_y, grid_x))

    # --- ポインティングベクトル成分の計算 ---
    for i in range(grid_y):
        for j in range(grid_x):
            Bx = Data[i * grid_x + j, 0] * mag
            By = Data[i * grid_x + j, 1] * mag
            Ez = Data[i * grid_x + j, 2] * mag
            Sx_Output[i, j] = (By * Ez) / mu_0
            Sy_Output[i, j] = (-Bx * Ez) / mu_0

    # --- プラズマ終了点以降の領域を抽出 ---
    plasma_end_x = int(np.abs((NV + NM) / 2))
    Sx_region = Sx_Output[:, plasma_end_x:]
    Sy_region = Sy_Output[:, plasma_end_x:]

    # --- 各点を角度ごとに合算 ---
    Sx_all = Sx_region.flatten()
    Sy_all = Sy_region.flatten()

    # 0除外
    valid_mask = (Sx_all != 0) | (Sy_all != 0)
    Sx_all = Sx_all[valid_mask]
    Sy_all = Sy_all[valid_mask]

    # 角度と強度
    theta = np.arctan2(Sy_all, Sx_all)  # [-π, π]
    theta_deg = (np.degrees(theta) + 360) % 360
    intensity = np.sqrt(Sx_all**2 + Sy_all**2)

    # --- 角度ごとにビン化（極座標用） ---
    bins = np.linspace(0, 360, 181)  # 2°刻みで滑らかに
    intensity_binned, _ = np.histogram(theta_deg, bins=bins, weights=intensity)
    angles = 0.5 * (bins[:-1] + bins[1:])

    # --- 正規化 ---
    intensity_norm = (
        intensity_binned / np.max(intensity_binned)
        if np.max(intensity_binned) > 0
        else intensity_binned
    )

    # --- プロット ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.radians(angles), intensity_norm, lw=2)
    ax.set_xticks(np.radians(np.arange(0, 360, 15)))
    ax.set_title(f"file={DataNum}", size="small", loc="right")

    # --- 時刻情報 ---
    real_time = DataNum * int_snap * wavelength * 1.0e-6 / 3.0e8 / ndav * 1.0e15
    time_step = int(DataNum * ndav)
    ax.set_title(f"{real_time:8.02f} fsec = {time_step} step", size="small", loc="left")
    # --- 時刻情報 ---
    real_time = DataNum * int_snap * wavelength * 1.0e-6 / 3.0e8 / ndav * 1.0e15
    time_step = int(DataNum * ndav)
    ax.set_title(f"{real_time:8.02f} fsec = {time_step} step", size="small", loc="left")

    intensity_max = np.max(intensity_binned)
    # --- FWHM・平均角度・標準偏差 ---
    if intensity_max > 0:
        half_max = 0.5
        mask_upper = (angles >= 0) & (angles <= 180)
        mask_lower = (angles >= 180) & (angles <= 360)

        angles_upper = angles[mask_upper]
        angles_lower = angles[mask_lower]
        intensity_upper = intensity_norm[mask_upper]
        intensity_lower = intensity_norm[mask_lower]

        active_upper = angles_upper[intensity_upper >= half_max]
        active_lower = angles_lower[intensity_lower >= half_max]

        width_upper = (
            (active_upper.max() - active_upper.min()) if active_upper.size > 0 else 0
        )
        width_lower = (
            (active_lower.max() - active_lower.min()) if active_lower.size > 0 else 0
        )
        theta_FWHM = width_upper + width_lower

        theta_rad = np.radians(angles)
        mean_angle_rad = np.angle(np.sum(intensity_norm * np.exp(1j * theta_rad)))
        theta_mean = np.degrees(mean_angle_rad) % 360

        delta = (angles - theta_mean + 180) % 360 - 180
        theta_std = np.sqrt(
            np.sum((delta**2) * intensity_norm) / np.sum(intensity_norm)
        )

        fwhm_text = (
            f"Upper={width_upper:.2f}°, Lower={width_lower:.2f}°, "
            f"Total={theta_FWHM:.2f}°, Mean={theta_mean:.2f}°, Std={theta_std:.2f}°"
        )
    else:
        fwhm_text = "データがすべてゼロです"

    fig.text(0.5, 0.02, fwhm_text, ha="center", va="bottom", fontsize=12, color="blue")

    # --- 保存 ---
    plt.savefig(f"{new_dir_path}/{title}_{DataNum:05d}.png")
    plt.close("all")

    del fig, ax, Data, Sx_Output, Sy_Output


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
        directionalPlot(
            snap_file, snap_value, new_dir_path
        )  # カラーマップ作成関数 directionalPlot を実行
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
            directionalPlot(
                datafile, DataNum, new_dir_path
            )  # カラーマップ作成関数 directionalPlot を実行
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
