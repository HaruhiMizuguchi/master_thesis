import os
import torch
import paramiko
import chardet
from config.secrets import SSH_PRIVATE_KEY_PATH

pc_host = "133.87.136.147"
pc_port = 22
pc_username = "mizuguchi haruhi"
private_key_path = SSH_PRIVATE_KEY_PATH
pc_dest_path = "C:/Users/mizuguchi haruhi/shuuronn/google_drive/tmp/tmp_model.pth"

def save_model_on_server(model, save_path):
    """
    サーバ側で学習済みモデルを torch.save する。
    例) save_path = "/tmp/round_1.pth"
    """
    # model が nn.Module の場合、state_dict() を保存する
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), save_path)
    else:
        # 既に state_dict 形式や OrderedDict の場合はそのまま保存
        torch.save(model, save_path)
    print(f"Model saved on server: {save_path}")

def upload_file_to_pc_key(
    server_file_path,
    pc_host,
    pc_port,
    pc_username,
    private_key_path,
    pc_dest_path,
    key_passphrase=None
):
    """
    サーバ側から、自分用PCへファイルをアップロードする (SFTPで送信, SSH鍵認証)

    Parameters:
    - server_file_path: サーバ側の転送元ファイルパス (例: "/tmp/round_1.pth")
    - pc_host: 自分用PCのホスト名またはIPアドレス
    - pc_port: SSH接続に使用するポート (通常は22)
    - pc_username: 自分用PCでSSH接続する際のユーザー名
    - private_key_path: サーバ側にある秘密鍵ファイルのパス (例: "/home/serveruser/.ssh/id_rsa")
    - pc_dest_path: 自分用PC側の保存先パス (例: "/home/username/received/round_1.pth")
    - key_passphrase: 秘密鍵にパスフレーズが設定されている場合、その文字列 (例: "my_secret_passphrase")
    """
    if not os.path.exists(server_file_path):
        raise FileNotFoundError(f"Server file not found: {server_file_path}")

     # 1. 鍵の読み込み
    try:
        # Ed25519形式の鍵を読み込む
        key = paramiko.Ed25519Key.from_private_key_file(private_key_path, password=key_passphrase)
    except paramiko.SSHException as e:
        raise ValueError(f"Could not load private key: {e}")

    # 2. Transport作成・接続
    transport = paramiko.Transport((pc_host, pc_port))
    transport.connect(username=pc_username, pkey=key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # 3. アップロード先のディレクトリを自動で作る (必要に応じて)
    remote_dir = os.path.dirname(pc_dest_path)
    # ディレクトリを一階層ずつチェックして無ければ作成
    original_dir = sftp.getcwd()
    for sub_dir in remote_dir.split('/'):
        if not sub_dir:
            continue
        try:
            sftp.chdir(sub_dir)
        except IOError:
            sftp.mkdir(sub_dir)
            sftp.chdir(sub_dir)

    # カレントディレクトリを元に戻す
    sftp.chdir(original_dir)

    # 4. ファイルをアップロード
    sftp.put(server_file_path, pc_dest_path)
    print(f"Uploaded file from {server_file_path} to {pc_dest_path} on PC.")

    # 5. クローズ
    sftp.close()
    transport.close()

def run_remote_command(
    pc_host,
    pc_port,
    pc_username,
    private_key_path,
    command,
    key_passphrase=None
):
    """
    サーバ側から、自分用PC上のコマンド(スクリプト)をSSH鍵認証で実行する。
    
    Parameters:
    - pc_host, pc_port, pc_username, private_key_path : 鍵認証情報
    - command : 実行したいコマンド文字列 (例: "python /home/pc_user/google_drive_upload.py --file /home/pc_user/received/round_1.pth")
    - key_passphrase : 秘密鍵がパスフレーズ付きなら指定
    """
     # 1. 鍵の読み込み
    try:
        # Ed25519形式の鍵を読み込む
        key = paramiko.Ed25519Key.from_private_key_file(private_key_path, password=key_passphrase)
    except paramiko.SSHException as e:
        raise ValueError(f"Could not load private key: {e}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=pc_host, port=pc_port, username=pc_username, pkey=key)
    
    stdin, stdout, stderr = ssh.exec_command(command)
    def decode_output(output):
        if not output:  # 出力が空の場合、そのまま空文字列を返す
            return ""
        result = chardet.detect(output)
        encoding = result.get('encoding', 'utf-8')  # エンコーディングがNoneの場合、'utf-8'をデフォルトにする
        return output.decode(encoding=encoding, errors="replace")
    out = decode_output(stdout.read())  # 標準出力
    err = decode_output(stderr.read())  # 標準エラー出力

    print("=== Remote command output ===")
    print(out)
    if err:
        print("=== Remote command error ===")
        print(err)

    ssh.close()

def save_model_drive_relay_pc(
    model,
    drive_folder,
    drive_file
    ):
    # 1. 学習済モデルをサーバ側に保存
    save_model_on_server(model, "/home/haruhi_mizuguchi/master_thesis/ADA/DiaNA/tmp/tmp_model.pth")
    # 2. サーバ→PCへファイルをアップロード
    server_file_path = "/home/haruhi_mizuguchi/master_thesis/ADA/DiaNA/tmp/tmp_model.pth"
    upload_file_to_pc_key(
        server_file_path=server_file_path,
        pc_host=pc_host,
        pc_port=pc_port,
        pc_username=pc_username,
        private_key_path=private_key_path,
        pc_dest_path=pc_dest_path,
        key_passphrase=None)
    # 3. PC側でGoogle DriveアップロードのPythonスクリプトを実行
    python_path = "C:/Users/mizuguchi haruhi/anaconda3/envs/diana/python.exe"
    command = (
        f'cd "C:/Users/mizuguchi haruhi/shuuronn/google_drive" && '
        f'"{python_path}" save_to_drive.py --file "{pc_dest_path}" --drive_folder "{drive_folder}" --drive_file "{drive_file}"'
    )
    run_remote_command(
        pc_host=pc_host,
        pc_port=pc_port,
        pc_username=pc_username,
        private_key_path=private_key_path,
        command=command,
        key_passphrase=None
    )

def download_file_from_pc_key(
    pc_host,
    pc_port,
    pc_username,
    private_key_path,
    pc_file_path,
    server_dest_path,
    key_passphrase=None
):
    """
    サーバ側から、自分用PCのファイルをダウンロードする (SFTPで取得, SSH鍵認証)

    Parameters:
    - pc_host: 自分用PCのホスト名またはIPアドレス
    - pc_port: SSH接続に使用するポート (通常は22)
    - pc_username: 自分用PCでSSH接続する際のユーザー名
    - private_key_path: サーバ側にある秘密鍵ファイルのパス
    - pc_file_path: 自分用PC上の取得元パス (例: "/home/username/received/round_2.pth")
    - server_dest_path: サーバ側の保存先パス (例: "/tmp/round_2_downloaded.pth")
    - key_passphrase: 秘密鍵にパスフレーズが設定されている場合、その文字列
    """
     # 1. 鍵の読み込み
    try:
        # Ed25519形式の鍵を読み込む
        key = paramiko.Ed25519Key.from_private_key_file(private_key_path, password=key_passphrase)
    except paramiko.SSHException as e:
        raise ValueError(f"Could not load private key: {e}")

    # 2. Transport作成・接続
    transport = paramiko.Transport((pc_host, pc_port))
    transport.connect(username=pc_username, pkey=key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # 3. サーバ側の保存先ディレクトリを作成しておく
    local_dir = os.path.dirname(server_dest_path)
    os.makedirs(local_dir, exist_ok=True)

    # 4. ファイルをダウンロード
    sftp.get(pc_file_path, server_dest_path)
    print(f"Downloaded file from PC:{pc_file_path} to Server:{server_dest_path}")

    # 5. クローズ
    sftp.close()
    transport.close()

def load_model_on_server(load_path, model_class=None, device='cpu'):
    """
    サーバ側で torch.load してモデルを復元する。
    
    - load_path: サーバ側のファイルパス (例: "/tmp/round_1_downloaded.pth")
    - model_class: モデルクラスを与えると、state_dict() を load して戻す
      (例: ResNet等をインスタンス化してから state_dict を読み込み)
    - device: "cpu" or "cuda" etc.

    Returns:
    - 復元されたモデル (model_class が None の場合は state_dict or そのままオブジェクト)
    """
    if model_class:
        model = model_class().to(device)
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded (with class) from {load_path}")
        return model
    else:
        obj = torch.load(load_path, map_location=device)
        print(f"Object loaded from {load_path}")
        return obj
