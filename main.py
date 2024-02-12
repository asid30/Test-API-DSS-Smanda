# library
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import operasi as op

# initiation instance FastAPI as app
app = FastAPI()


# base model from point item
class PoinItem(BaseModel):
    poin_bio: int
    poin_fis: int
    poin_kim: int
    poin_it: int
    poin_mtkl: int
    poin_sos: int
    poin_eko: int
    poin_geo: int
    poin_fra: int
    poin_jpn: int


# base model from interest calculation
class MinatInput(BaseModel):
    PAI: float
    Pendidikan_Pancasila: float
    Bahasa_Indonesia: float
    MTK: float
    Bahasa_Inggris: float
    Penjaskes: float
    Sejarah: float
    Seni_Budaya: float
    pilihan1: int
    pilihan2: int
    pilihan3: int


# base model from competition calculation
class WajibInput(BaseModel):
    PAI: float
    Pendidikan_Pancasila: float
    Bahasa_Indonesia: float
    MTK: float
    Bahasa_Inggris: float
    Penjaskes: float
    Sejarah: float
    Seni_Budaya: float


# Columns and index
columns = op.mapel_wajib()
subjects = op.mapel_minat()


# return MPP as list
@app.get("/mapel_minat")
def read_root():
    return op.mapel_minat()


# return MPW as list
@app.get("/mapel_wajib")
def read_root():
    return op.mapel_wajib()


# take index and return as MPP
@app.post("/get_mpp")
def get_mpp(
    pilihan1: float = Form(...),
    pilihan2: float = Form(...),
    pilihan3: float = Form(...),
):
    daftar = []
    daftar.append(op.index_to_mapel_minat(int(pilihan1) - 1))
    daftar.append(op.index_to_mapel_minat(int(pilihan2) - 1))
    daftar.append(op.index_to_mapel_minat(int(pilihan3) - 1))
    return daftar


# taking competition result from web and return it as list and it will used for system
@app.post("/kompetensi")
def proses_nilai(data: WajibInput, items: List[PoinItem]):
    # Dapat mengakses nilai menggunakan objek data
    PAI = data.PAI
    Pendidikan_Pancasila = data.Pendidikan_Pancasila
    Bahasa_Indonesia = data.Bahasa_Indonesia
    MTK = data.MTK
    Bahasa_Inggris = data.Bahasa_Inggris
    Penjaskes = data.Penjaskes
    Sejarah = data.Sejarah
    Seni_Budaya = data.Seni_Budaya

    ds = op.data_nilai_siswa_input(
        PAI,
        Pendidikan_Pancasila,
        Bahasa_Indonesia,
        MTK,
        Bahasa_Inggris,
        Penjaskes,
        Sejarah,
        Seni_Budaya,
    )
    data_list = [item.dict() for item in items]
    df = pd.DataFrame(data_list, index=columns)
    df = df.transpose()
    df = df.set_index(pd.Index(subjects))
    df = op.tabel_korelasi_normalisasi(df)
    # df = op.tabel_korelasi_normalisasi()

    hasil_perkalian_df_dan_ds = op.tabel_kompetensi(df, ds)
    return op.dataframe_to_dict_converter(hasil_perkalian_df_dan_ds)


# taking interests result from web and return it as list and it will used for system
@app.post("/minat")
def proses_nilai(data: MinatInput, items: List[PoinItem]):
    # Dapat mengakses nilai menggunakan objek data
    pilihan1 = data.pilihan1
    pilihan2 = data.pilihan2
    pilihan3 = data.pilihan3
    PAI = data.PAI
    Pendidikan_Pancasila = data.Pendidikan_Pancasila
    Bahasa_Indonesia = data.Bahasa_Indonesia
    MTK = data.MTK
    Bahasa_Inggris = data.Bahasa_Inggris
    Penjaskes = data.Penjaskes
    Sejarah = data.Sejarah
    Seni_Budaya = data.Seni_Budaya

    ds = op.data_nilai_siswa_input(
        PAI,
        Pendidikan_Pancasila,
        Bahasa_Indonesia,
        MTK,
        Bahasa_Inggris,
        Penjaskes,
        Sejarah,
        Seni_Budaya,
    )
    data_list = [item.dict() for item in items]
    df = pd.DataFrame(data_list, index=columns)
    df = df.transpose()
    df = df.set_index(pd.Index(subjects))
    df = op.tabel_korelasi_normalisasi(df)
    # df = op.tabel_korelasi_normalisasi()

    hasil_perkalian_df_dan_ds = op.tabel_kompetensi(df, ds)
    hasil_prediksi_dengan_minat = op.tabel_kompetensi_minat(
        [pilihan1, pilihan2, pilihan3], hasil_perkalian_df_dan_ds
    )
    return op.dataframe_to_dict_converter(hasil_prediksi_dengan_minat)


# testing
@app.get("/test")
def prediksi():
    ds = op.data_nilai_siswa_random()
    df = op.tabel_korelasi_normalisasi()
    hasil_perkalian_df_dan_ds = op.tabel_kompetensi(df, ds)
    return op.dataframe_to_dict_converter(hasil_perkalian_df_dan_ds)


# to start API
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
