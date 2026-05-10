用途：把微信里保存的 submission CSV 复制到本目录后，可在工程根运行：
  python tools/compare_glevel_submissions.py external/submissions_peer/*.csv --vote-out external/submissions_peer/submission_vote_majority.csv

期望列：id + g_level_pred（或 g_level）；编码 0/1/2 或 1/2/3 均可。

--- 从 Windows 拷到本服务器（任选一种）---

【方式 A】PowerShell / CMD 里用 scp（需已安装 OpenSSH 客户端，并把 USER 与 SERVER 换成你的 SSH 登录）

  cd "C:\Users\AntonyTang\Documents\xwechat_files\wxid_pkf7kk1e6qxa22_3b4a\msg\file\2026-05"

  scp submission1_0.53077.csv submission2_0.50769.csv submission_0.53846.csv submission5_0.55385.csv ^
    USER@SERVER:/home/emo/antonytang/AVI2026_Track2_GLevel/external/submissions_peer/

【方式 B】WinSCP / FileZilla：SFTP 登录服务器，拖到上述远程路径即可。

【方式 C】若你用 Cursor Remote-SSH 已打开服务器工程：在 Windows 资源管理器复制 4 个文件，
  粘贴到 Cursor 左侧文件树里的 external/submissions_peer/（若未显示则右键 Refresh）。

拷完后在服务器上检查：
  ls -la /home/emo/antonytang/AVI2026_Track2_GLevel/external/submissions_peer/
