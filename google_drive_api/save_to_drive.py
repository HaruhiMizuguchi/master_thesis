# 必要なライブラリのインストール
# pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import time

# Google Drive APIの設定
SCOPES = ['https://www.googleapis.com/auth/drive.file']  # Driveのファイル操作用スコープ
CLIENT_SECRET_FILE = "client_secret_881044251580-8pip2gv64qdjhrqcqbhco46ri28nteri.apps.googleusercontent.com.json"  # OAuth 2.0クライアントのJSONファイルパスを指定

# OAuth認証を行う
def authenticate_with_user_account():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

service = authenticate_with_user_account()

# アップロードするフォルダのパスを指定
local_folder_path = '/home/haruhi_mizuguchi/master_thesis/ADA/DiaNA/with_gmm_only_tgt_sup_models_fix_seed_not_augment_ResNet34'  # アップロードしたいフォルダのパス
google_drive_folder_id = ''  # アップロード先のGoogle DriveフォルダID

# Google Drive上にフォルダを作成する関数
def create_folder_in_drive(folder_name, parent_id):
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

# フォルダ構造を再現するための関数
def create_drive_folders_structure(local_folder_path, google_drive_folder_id):
    folder_id_map = {local_folder_path: google_drive_folder_id}
    for root, dirs, _ in os.walk(local_folder_path):
        for directory in dirs:
            local_dir_path = os.path.join(root, directory)
            parent_folder_id = folder_id_map[root]
            drive_folder_id = create_folder_in_drive(directory, parent_folder_id)
            folder_id_map[local_dir_path] = drive_folder_id
    return folder_id_map

# ファイルを一つずつアップロードする関数
def upload_files_one_by_one(local_folder_path, folder_id_map):
    for root, _, files in os.walk(local_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            parent_folder_id = folder_id_map[root]
            try:
                file_metadata = {
                    'name': os.path.basename(file_path),
                    'parents': [parent_folder_id]
                }
                media = MediaFileUpload(file_path, resumable=True)
                file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                print(f"Uploaded {file_path} to Google Drive with File ID: {file.get('id')}")
                time.sleep(1)  # 負荷を軽減するために1秒の待機時間を設定
            except Exception as e:
                print(f"Failed to upload {file_path}: {e}")

# フォルダ構造を先に再現し、ファイルを一つずつアップロードする
folder_id_map = create_drive_folders_structure(local_folder_path, google_drive_folder_id)
upload_files_one_by_one(local_folder_path, folder_id_map)
