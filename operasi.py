# library
import numpy as np
import pandas as pd
import warnings

# supress warning
warnings.simplefilter(action="ignore", category=FutureWarning)


# default correlation table
def korelasi():
    data = {
        "PAI": [1, 1, 1, 1, 1, 3, 1, 1, 1, 1],
        "Pendidikan Pancasila": [1, 1, 1, 1, 1, 5, 1, 3, 1, 1],
        "Bahasa Indonesia": [3, 1, 2, 1, 1, 3, 2, 3, 1, 1],
        "MTK": [2, 5, 5, 5, 5, 1, 5, 1, 1, 1],
        "Bahasa Inggris": [3, 3, 3, 5, 2, 2, 2, 2, 1, 1],
        "Penjaskes": [4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        "Sejarah": [1, 1, 1, 1, 1, 5, 4, 4, 1, 1],
        "Seni Budaya": [1, 1, 1, 1, 1, 4, 2, 3, 1, 1],
    }
    return data


# custom correlation table
def set_data_korelasi(
    Biologi: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Fisika: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Kimia: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Informatika: list = [1, 1, 1, 1, 1, 1, 1, 1],
    MTK_Lanjut: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Sosiologi: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Ekonomi: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Geografi: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Bahasa_Prancis: list = [1, 1, 1, 1, 1, 1, 1, 1],
    Bahasa_Jepang: list = [1, 1, 1, 1, 1, 1, 1, 1],
):
    data = [
        Biologi,
        Fisika,
        Kimia,
        Informatika,
        MTK_Lanjut,
        Sosiologi,
        Ekonomi,
        Geografi,
        Bahasa_Prancis,
        Bahasa_Jepang,
    ]

    return data


# function for taking index return MPP
def index_to_mapel_minat(pilihan):
    temp = mapel_minat()
    return temp[pilihan]


# function for list of MPW
def mapel_wajib():
    columns = [
        "PAI",  # 1
        "Pendidikan Pancasila",  # 2
        "Bahasa Indonesia",  # 3
        "MTK",  # 4
        "Bahasa Inggris",  # 5
        "Penjaskes",  # 6
        "Sejarah",  # 7
        "Seni Budaya",  # 8
    ]
    return columns


# function for list of MPP
def mapel_minat():
    subjects = [
        "Biologi",  # 1
        "Fisika",  # 2
        "Kimia",  # 3
        "Informatika",  # 4
        "MTK Lanjut",  # 5
        "Sosiologi",  # 6
        "Ekonomi",  # 7
        "Geografi",  # 8
        "Bahasa Prancis",  # 9
        "Bahasa Jepang",  # 10
    ]
    return subjects


# function for making correlation table to pandas
def tabel_korelasi(data=korelasi(), columns=mapel_wajib(), index=mapel_minat()):
    return pd.DataFrame(data=data, columns=columns, index=index)


# function for input random testing
def data_nilai_siswa_random(index=mapel_minat()):
    data_nilai = {
        "PAI": np.random.randint(30, 101),
        "Pendidikan Pancasila": np.random.randint(30, 101),
        "Bahasa Indonesia": np.random.randint(30, 101),
        "MTK": np.random.randint(30, 101),
        "Bahasa Inggris": np.random.randint(30, 101),
        "Penjaskes": np.random.randint(30, 101),
        "Sejarah": np.random.randint(30, 101),
        "Seni Budaya": np.random.randint(30, 101),
    }
    return pd.DataFrame(data=data_nilai, index=index)


# function for convert nilai siswa into pandas
def data_nilai_siswa_input(
    PAI: int,
    Pendidikan_Pancasila: int,
    Bahasa_Indonesia: int,
    MTK: int,
    Bahasa_Inggris: int,
    Penjaskes: int,
    Sejarah: int,
    Seni_Budaya: int,
    index=mapel_minat(),
):
    data_nilai = {
        "PAI": PAI,
        "Pendidikan Pancasila": Pendidikan_Pancasila,
        "Bahasa Indonesia": Bahasa_Indonesia,
        "MTK": MTK,
        "Bahasa Inggris": Bahasa_Inggris,
        "Penjaskes": Penjaskes,
        "Sejarah": Sejarah,
        "Seni Budaya": Seni_Budaya,
    }
    return pd.DataFrame(data=data_nilai, index=index)


# function for normalization correlation table
def tabel_korelasi_normalisasi(DataFrame=tabel_korelasi()):
    jumlah_nilai = []
    for j in range(len(DataFrame)):
        jumlah = 0

        for i in range(DataFrame.shape[1]):
            jumlah += DataFrame.iloc[j][i]

        data_sekarang = DataFrame.iloc[j] / jumlah
        DataFrame.iloc[j] = data_sekarang
        jumlah_nilai.append(jumlah)
    return DataFrame


# function for calculation for each nilai siswa to correlation point
def tabel_kompetensi(DataFrame1, DataFrame2):
    # Perkalian 2 tabel dengan ordo yang sama
    hasil = DataFrame1 * DataFrame2
    return hasil_prediksi(hasil)


# function for convert prediction point into table
def hasil_prediksi(DataFrame):
    jumlah_bobot_nilai = []
    for j in range(len(DataFrame)):
        bbt = 0

        for i in range(DataFrame.shape[1]):
            bbt += DataFrame.iloc[j][i]
            round(bbt, 2)

        jumlah_bobot_nilai.append(bbt)
    # Mengurangi sebagian dari nilai di indeks 8 dan 9 untuk mata pelajaran bahasa asing
    jumlah_bobot_nilai[8] -= jumlah_bobot_nilai[8] * 0.1
    jumlah_bobot_nilai[9] -= jumlah_bobot_nilai[9] * 0.1
    DataFrame["total_prediksi_terbaik"] = jumlah_bobot_nilai

    # sorting DESC
    return round(
        DataFrame.sort_values(by=["total_prediksi_terbaik"], ascending=False), 2
    )


# function from pandas into dict
def dataframe_to_dict_converter(DataFrame):
    top = DataFrame
    mp = top.index.tolist()
    top["mata_pelajaran"] = mp
    top = top.set_axis([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], axis=0)
    hasil = top[["mata_pelajaran", "total_prediksi_terbaik"]].to_dict()
    return hasil


# function for calculate prediction according to students interests
def tabel_kompetensi_minat(pilihan: list, DataFrame):
    subjects = [
        "Biologi",  # 1
        "Fisika",  # 2
        "Kimia",  # 3
        "Informatika",  # 4
        "MTK Lanjut",  # 5
        "Sosiologi",  # 6
        "Ekonomi",  # 7
        "Geografi",  # 8
        "Bahasa Prancis",  # 9
        "Bahasa Jepang",  # 10
    ]
    mapel = ["Tidak ada", "Tidak ada", "Tidak ada"]
    for i in range(3):
        mapel_index = int(
            pilihan[i] - 1
        )  # Mendapatkan indeks mata pelajaran berdasarkan pilihan input
        mapel[i] = (
            subjects[mapel_index] if 0 <= mapel_index < len(subjects) else "Tidak ada"
        )
    # first choice add 30% total points
    DataFrame = DataFrame.T
    DataFrame["{ngodong}".format(ngodong=mapel[0])]["total_prediksi_terbaik"] += (
        DataFrame["{ngodong}".format(ngodong=mapel[0])]["total_prediksi_terbaik"] * 0.3
    )
    # first choice add 20% total points
    DataFrame["{ngodong}".format(ngodong=mapel[1])]["total_prediksi_terbaik"] += (
        DataFrame["{ngodong}".format(ngodong=mapel[1])]["total_prediksi_terbaik"] * 0.2
    )
    # first choice add 10% from total points
    DataFrame["{ngodong}".format(ngodong=mapel[2])]["total_prediksi_terbaik"] += (
        DataFrame["{ngodong}".format(ngodong=mapel[2])]["total_prediksi_terbaik"] * 0.1
    )
    DataFrame = DataFrame.T
    # sorting DESC
    return round(
        DataFrame.sort_values(by=["total_prediksi_terbaik"], ascending=False), 2
    )
