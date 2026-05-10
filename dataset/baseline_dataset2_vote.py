import os
import sys
import numpy as np

# 训练入口会打印；若服务器仍报 _resolve_feature_path，说明本文件未同步到「实际跑 Python 的那台机器」
FEATURE_LOADER_REVISION = 6  # 空集时 stderr 诊断首条样本缺哪一模态
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import ast

try:
    import soundfile as sf
except ImportError:
    sf = None


def _wav_duration_sec(path):
    if sf is None or not os.path.isfile(path):
        return 1.0
    try:
        info = sf.info(path)
        return max(float(info.frames) / float(info.samplerate), 1e-6)
    except Exception:
        return 1.0


def encode_g_level(raw, int_encoding="zero"):
    """将 g_level 转为类别索引 0/1/2（CrossEntropy 用长整型目标，勿用 one-hot）。

    int_encoding:
      - ``one``（赛方官方）：CSV 整数标签为 1/2/3 → 映射为内部 0/1/2。
      - ``zero``：CSV 已为 0/1/2（与内部类下标一致）。
      字符串 low/medium/high 及数字字符串在两模式下解析方式一致。
    """
    enc = (int_encoding or "zero").strip().lower()
    if enc not in ("zero", "one"):
        raise ValueError(f"g_level int_encoding must be 'zero' or 'one', got {int_encoding!r}")

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        raise ValueError("g_level is missing or NaN")
    if isinstance(raw, (bool, np.bool_)):
        raw = int(raw)
    if isinstance(raw, (float, np.floating)) and float(raw).is_integer():
        return encode_g_level(int(raw), int_encoding=enc)
    if isinstance(raw, (int, np.integer)):
        v = int(raw)
        if enc == "one":
            if v in (1, 2, 3):
                return v - 1
            raise ValueError(
                f"g_level integer {v} invalid for int_encoding=one (expect 1–3 = Low/Medium/High)"
            )
        if v in (0, 1, 2):
            return v
        raise ValueError(
            f"g_level integer {v} invalid for int_encoding=zero (expect 0–2); "
            "若赛方使用 1–3 请设 --g_level_int_encoding one"
        )
    s = str(raw).strip().lower()
    if s.isdigit():
        return encode_g_level(int(s), int_encoding=enc)
    mapping = {
        "low": 0,
        "medium": 1,
        "mid": 1,
        "med": 1,
        "high": 2,
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown g_level label: {raw!r}")


def _list_npy_filenames(dir_path):
    """目录内所有 .npy 文件名（小写扩展名匹配）；每目录 listdir 一次。"""
    if not dir_path or not os.path.isdir(dir_path):
        return []
    return [fn for fn in os.listdir(dir_path) if fn.lower().endswith(".npy")]


def _load_feat_norm_bundle(args) -> dict | None:
    """由 train_task2_glevel 的 feat_norm_npz / feat_norm_apply 加载每模态 (mu, std)。"""
    if args is None:
        return None
    path = (getattr(args, "feat_norm_npz", None) or "").strip()
    mode = (getattr(args, "feat_norm_apply", "none") or "none").strip().lower()
    if mode != "all" or not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"feat_norm_npz 不存在: {path}")
    z = np.load(path)
    eps = float(getattr(args, "feat_norm_eps", 1e-6))
    bundle = {}
    for k in ("audio", "video", "text"):
        mu = np.asarray(z[f"{k}_mu"], dtype=np.float32)
        std = np.asarray(z[f"{k}_std"], dtype=np.float32) + eps
        bundle[k] = (mu, std)
    return bundle


def _apply_feat_norm_numpy(features: dict, bundle: dict | None) -> dict:
    if not bundle:
        return features
    out = {}
    for key, arr in features.items():
        if key in bundle:
            mu, std = bundle[key]
            x = arr.astype(np.float32, copy=False)
            out[key] = ((x - mu) / std).astype(np.float32)
        else:
            out[key] = arr
    return out


def _pick_npy_path(base_dir, filenames, sample_id, q):
    """prefix = {id}_{q}，对缓存的文件名做 startswith 匹配（与赛方命名一致）。"""
    if not base_dir or not filenames:
        return None
    sid = str(sample_id).strip()
    prefix = f"{sid}_{q}"
    hits = [fn for fn in filenames if fn.startswith(prefix) and fn.lower().endswith(".npy")]
    if not hits:
        return None
    return os.path.join(base_dir, sorted(hits)[0])


def _resolve_from_name_lists(
    sample_id,
    q,
    primary_dir,
    fallback_dir,
    names_p,
    names_f,
    fallback2_dir=None,
    names_f2=None,
):
    p = _pick_npy_path(primary_dir, names_p, sample_id, q)
    if p is not None:
        return p
    if fallback_dir and names_f is not None:
        p = _pick_npy_path(fallback_dir, names_f, sample_id, q)
        if p is not None:
            return p
    if fallback2_dir and names_f2 is not None:
        return _pick_npy_path(fallback2_dir, names_f2, sample_id, q)
    return None


def _row_has_all_features_cached(
    sample_id,
    question,
    audio_dir,
    video_dir,
    text_dir,
    fa,
    fv,
    ft,
    la,
    lab,
    lv,
    lvb,
    lt,
    ltb,
    modalities_required=None,
    *,
    fa2=None,
    lab2=None,
    fv2=None,
    lvb2=None,
    ft2=None,
    ltb2=None,
):
    """modalities_required：需要的模态名集合，如 frozenset({'text'})；None 表示 audio/video/text 均需 .npy。"""
    mods = modalities_required
    if mods is None:
        mods = frozenset({"audio", "video", "text"})
    for q in question:
        if "audio" in mods and _resolve_from_name_lists(sample_id, q, audio_dir, fa, la, lab, fa2, lab2) is None:
            return False
        if "video" in mods and _resolve_from_name_lists(sample_id, q, video_dir, fv, lv, lvb, fv2, lvb2) is None:
            return False
        if "text" in mods and _resolve_from_name_lists(sample_id, q, text_dir, ft, lt, ltb, ft2, ltb2) is None:
            return False
    return True


def _drop_rows_missing_features(
    df,
    question,
    audio_dir,
    video_dir,
    text_dir,
    fallback_audio_dir,
    fallback_video_dir,
    fallback_text_dir,
    split_name,
    la,
    lab,
    lv,
    lvb,
    lt,
    ltb,
    allow_empty=False,
    modalities_required=None,
    *,
    fallback_audio_dir_2=None,
    lab2=None,
    fallback_video_dir_2=None,
    lvb2=None,
    fallback_text_dir_2=None,
    ltb2=None,
):
    """剔除缺少任一题、任一模态 .npy 的行（与 __getitem__ 解析规则一致）。"""
    n0 = len(df)
    keep = []
    dropped_ids = []
    for _, row in df.iterrows():
        sid = row["id"]
        if _row_has_all_features_cached(
            sid,
            question,
            audio_dir,
            video_dir,
            text_dir,
            fallback_audio_dir,
            fallback_video_dir,
            fallback_text_dir,
            la,
            lab,
            lv,
            lvb,
            lt,
            ltb,
            modalities_required=modalities_required,
            fa2=fallback_audio_dir_2,
            lab2=lab2,
            fv2=fallback_video_dir_2,
            lvb2=lvb2,
            ft2=fallback_text_dir_2,
            ltb2=ltb2,
        ):
            keep.append(True)
        else:
            keep.append(False)
            dropped_ids.append(str(sid).strip())
    out = df.loc[keep].reset_index(drop=True)
    n_drop = n0 - len(out)
    if n_drop:
        uniq = list(dict.fromkeys(dropped_ids))
        preview = ", ".join(uniq[:10])
        more = f" 等共 {len(uniq)} 个 id" if len(uniq) > 10 else ""
        print(
            f"[{split_name}] 特征不完整，已剔除 {n_drop}/{n0} 行（缺某题 audio/video/text 之一，"
            f"且回退目录仍无文件）。示例 id: {preview}{more}",
            file=sys.stderr,
        )
    if len(out) == 0:
        if allow_empty:
            print(
                f"[{split_name}] 过滤后无剩余样本（allow_empty）。"
                "测试阶段将跳过 predict；请补全 FEAT_TEST 或测试特征提取。",
                file=sys.stderr,
            )
            return out
        first_fail = None
        _mods = modalities_required if modalities_required is not None else frozenset({"audio", "video", "text"})
        if n0 > 0:
            sid0 = str(df.iloc[0]["id"]).strip()
            for q in question:
                if "audio" in _mods and _resolve_from_name_lists(
                    sid0, q, audio_dir, fallback_audio_dir, la, lab, fallback_audio_dir_2, lab2
                ) is None:
                    first_fail = ("audio", q)
                    break
                if "video" in _mods and _resolve_from_name_lists(
                    sid0, q, video_dir, fallback_video_dir, lv, lvb, fallback_video_dir_2, lvb2
                ) is None:
                    first_fail = ("video", q)
                    break
                if "text" in _mods and _resolve_from_name_lists(
                    sid0, q, text_dir, fallback_text_dir, lt, ltb, fallback_text_dir_2, ltb2
                ) is None:
                    first_fail = ("text", q)
                    break
            prefix0 = f"{sid0}_"
            text_hits = [fn for fn in lt if fn.startswith(prefix0)][:6]
            print(
                f"[{split_name}] 诊断（首条样本 id={sid0!r}）: "
                f"最先无法解析的模态={first_fail!r}；"
                f"audio 目录存在={bool(audio_dir and os.path.isdir(audio_dir))} .npy数={len(la)}；"
                f"video 目录存在={bool(video_dir and os.path.isdir(video_dir))} .npy数={len(lv)}；"
                f"text 目录存在={bool(text_dir and os.path.isdir(text_dir))} .npy数={len(lt)}；"
                f"text 回退目录={fallback_text_dir!r} 回退.npy数={len(ltb) if ltb is not None else 0}；"
                f"该 id 在 text 主目录下匹配前缀的文件示例={text_hits}",
                file=sys.stderr,
            )
            if first_fail and first_fail[0] == "text" and len(lt) == 0:
                print(
                    "[dataset] 提示: text 主目录下无任何 .npy。若仅用 Nanbeige 试跑目录，请 "
                    "export NANBEIGE_TEXT_SUBDIR=text_nb_smoke（见 vote_train_glevel.sh）"
                    " 或手动 export TEXT_TRAIN_DIR=.../text_nb_smoke；全量训练需提取到与 CSV 全 id 对齐的目录。",
                    file=sys.stderr,
                )
        extra = ""
        if "val" in split_name.lower() and first_fail and first_fail[0] == "text":
            extra = (
                " 常见原因（Nanbeige 试跑）：train_feature/text_nb_smoke 只有少量 train id，"
                "官方 val.csv 的 id 不在其中。请对验证集转写目录执行 extract（OUT 指向 "
                "val_feature/text_nb_smoke 等），再 export TEXT_VAL_DIR 指向该目录；"
                "或先全量提取再训练。"
            )
        raise ValueError(
            f"[{split_name}] 过滤后无剩余样本。请补全特征提取或检查 CSV / 特征目录。{extra}"
        )
    return out


def compute_hand_feats_row(transcript, audio_duration_sec):
    transcript = transcript if isinstance(transcript, str) else ""
    num_words = len(transcript.split())
    unique_ratio = len(set(transcript.lower().split())) / max(num_words, 1)
    if "." in transcript:
        sents = [s.strip() for s in transcript.split(".") if s.strip()]
        if sents:
            avg_sent_len = float(np.mean([len(s.split()) for s in sents]))
        else:
            avg_sent_len = float(num_words)
    else:
        avg_sent_len = float(num_words)
    speech_rate = num_words / max(float(audio_duration_sec), 1e-6)
    return np.array([num_words, unique_ratio, avg_sent_len, speech_rate], dtype=np.float32)

class MultimodalDatasetForTrainT2(Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        video_dir,
        text_dir,
        question,
        label_col,
        rating_csv,
        args=None,
        fallback_audio_dir=None,
        fallback_video_dir=None,
        fallback_text_dir=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir  # Directory containing audio features
        self.video_dir = video_dir  # Directory containing video features
        self.text_dir = text_dir    # Directory containing text features
        self.fallback_audio_dir = fallback_audio_dir
        self.fallback_video_dir = fallback_video_dir
        self.fallback_text_dir = fallback_text_dir
        self.question = question
        if args and getattr(args, "modalities", None):
            self.training_modal = frozenset(str(m).strip() for m in args.modalities)
        else:
            self.training_modal = frozenset({"audio", "video", "text"})
        self.audio_dim = int(getattr(args, "audio_dim", 512)) if args else 512
        self.video_dim = int(getattr(args, "video_dim", 512)) if args else 512
        self.text_dim = int(getattr(args, "text_dim", 768)) if args else 768
        self.use_hand = not (args and getattr(args, "no_hand", False))
        self._feat_norm = _load_feat_norm_bundle(args)
        # 缓存各目录 .npy 文件名；过滤与 __getitem__ 均用 startswith("{id}_{q}") 规则
        self._npy_audio = _list_npy_filenames(self.audio_dir)
        self._npy_audio_fb = _list_npy_filenames(self.fallback_audio_dir)
        self._npy_video = _list_npy_filenames(self.video_dir)
        self._npy_video_fb = _list_npy_filenames(self.fallback_video_dir)
        self._npy_text = _list_npy_filenames(self.text_dir)
        self._npy_text_fb = _list_npy_filenames(self.fallback_text_dir)
        _nodrop = args is not None and getattr(args, "no_drop_incomplete_features", False)
        if not _nodrop:
            self.data = _drop_rows_missing_features(
                self.data,
                self.question,
                self.audio_dir,
                self.video_dir,
                self.text_dir,
                self.fallback_audio_dir,
                self.fallback_video_dir,
                self.fallback_text_dir,
                split_name=f"train/val:{os.path.basename(csv_file)}",
                la=self._npy_audio,
                lab=self._npy_audio_fb,
                lv=self._npy_video,
                lvb=self._npy_video_fb,
                lt=self._npy_text,
                ltb=self._npy_text_fb,
                modalities_required=self.training_modal,
            )
        self.label_col = label_col
        self.rating = pd.read_csv(rating_csv)
        self.result_dict = {row['id']: row for _, row in self.rating.iterrows()}
        self.transcript_dir = getattr(args, "transcript_dir", None) if args else None
        self.wav_dir = getattr(args, "wav_dir", None) if args else None
        if self.transcript_dir == "":
            self.transcript_dir = None
        if self.wav_dir == "":
            self.wav_dir = None
        self.classification = getattr(args, "classification", False) if args else False
        self.glevel_dict = None
        gcsv = (getattr(args, "glevel_csv", None) or "").strip() if args else ""
        if self.classification:
            col = self.label_col[0]
            need_ids = set(self.data["id"].astype(str))
            label_map = {}
            labels_split = getattr(args, "labels_in_split_csv", False)

            if labels_split and col in self.data.columns:
                for _, row in self.data.iterrows():
                    label_map[str(row["id"])] = row[col]

            if gcsv:
                gcsv_path = os.path.abspath(gcsv)
                if not os.path.isfile(gcsv_path):
                    raise FileNotFoundError(
                        f"找不到 glevel_csv: {gcsv_path}\n"
                        f"请创建含 id 与 {col!r} 的 CSV，或去掉 --glevel_csv。"
                    )
                gf = pd.read_csv(gcsv_path)
                if "id" not in gf.columns:
                    raise ValueError(f"glevel_csv 须含列 id，当前为: {list(gf.columns)}")
                if col not in gf.columns:
                    raise ValueError(
                        f"glevel_csv 须含标签列 '{col}'，当前为: {list(gf.columns)}"
                    )
                for _, row in gf.iterrows():
                    sid = str(row["id"])
                    if sid in need_ids and sid not in label_map:
                        label_map[sid] = row[col]

            missing = need_ids - set(label_map.keys())
            if missing and col in self.rating.columns:
                id_to_row = {str(r["id"]): r for _, r in self.rating.iterrows()}
                for sid in list(missing):
                    if sid in id_to_row:
                        label_map[sid] = id_to_row[sid][col]
                missing = need_ids - set(label_map.keys())

            if missing:
                hint = (
                    "可：① 在划分 csv 中为每条样本增加 g_level；"
                    "② 使用 --glevel_csv（仅含缺失 id 或含 train+val 全集均可，只使用本划分中出现的 id）；"
                    "③ 在 rating_csv 中合并 g_level 列。"
                )
                if labels_split and col not in self.data.columns and not gcsv:
                    hint = (
                        f"划分文件 {csv_file} 无列 {col!r}，且未提供 --glevel_csv。"
                        "请上传带 g_level 的 val 表，或提供仅含验证集 id 的标签 CSV："
                        "  --glevel_csv ./data/glevel_val_supplement.csv"
                    )
                raise ValueError(
                    f"本划分共 {len(need_ids)} 个 id，其中 {len(missing)} 个缺少标签 {col!r}。"
                    f"示例: {list(sorted(missing))[:6]}\n{hint}"
                )

            if not label_map:
                raise ValueError(
                    "未解析到任何分类标签。请设置 --labels_in_split_csv、--glevel_csv，"
                    "或保证 rating_csv 含标签列。"
                )
            self.glevel_dict = label_map
            self._glevel_int_enc = str(
                getattr(args, "g_level_int_encoding", "zero") or "zero"
            ).lower()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_id = self.data.iloc[idx]['id']

        audio_paths = []
        video_paths = []
        text_paths = []
        features = {}

        for q in self.question:
            ap = vp = tp = None
            if "audio" in self.training_modal:
                ap = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.audio_dir,
                    self.fallback_audio_dir,
                    self._npy_audio,
                    self._npy_audio_fb,
                )
            if "video" in self.training_modal:
                vp = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.video_dir,
                    self.fallback_video_dir,
                    self._npy_video,
                    self._npy_video_fb,
                )
            if "text" in self.training_modal:
                tp = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.text_dir,
                    self.fallback_text_dir,
                    self._npy_text,
                    self._npy_text_fb,
                )
            if "audio" in self.training_modal and ap is None:
                tried_a = self.audio_dir
                if self.fallback_audio_dir:
                    tried_a = f"{self.audio_dir} 与 {self.fallback_audio_dir}"
                raise FileNotFoundError(
                    f"Missing audio for {sample_id}_{q}（已查找: {tried_a}）。"
                    "若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            if "video" in self.training_modal and vp is None:
                raise FileNotFoundError(
                    f"Missing video for {sample_id}_{q}。若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            if "text" in self.training_modal and tp is None:
                raise FileNotFoundError(
                    f"Missing text for {sample_id}_{q}。若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            audio_paths.append(ap)
            video_paths.append(vp)
            text_paths.append(tp)

        nq = len(self.question)
        # 不再平均池化，而是拼接为 sequence；未选用的模态以全零填充，保证 DataLoader 键齐全
        if "audio" in self.training_modal:
            features["audio"] = np.concatenate(
                [np.expand_dims(np.load(p), axis=0) for p in audio_paths], axis=0
            )
        else:
            features["audio"] = np.zeros((nq, self.audio_dim), dtype=np.float32)

        if "video" in self.training_modal:
            features["video"] = np.stack([np.load(p) for p in video_paths], axis=0)
        else:
            features["video"] = np.zeros((nq, self.video_dim), dtype=np.float32)

        if "text" in self.training_modal:
            features["text"] = np.stack([np.load(p) for p in text_paths], axis=0)
        else:
            features["text"] = np.zeros((nq, self.text_dim), dtype=np.float32)

        if self.use_hand:
            hand_rows = []
            for ap, vp, tp in zip(audio_paths, video_paths, text_paths):
                ref = tp or ap or vp
                if ref is None:
                    hand_rows.append(compute_hand_feats_row("", 1.0))
                    continue
                af = os.path.basename(ap) if ap else os.path.basename(ref)
                tf = os.path.basename(tp) if tp else os.path.basename(ref)
                transcript = ""
                duration = 1.0
                if self.transcript_dir:
                    base = os.path.splitext(tf)[0]
                    tpath = os.path.join(self.transcript_dir, base + ".txt")
                    if os.path.isfile(tpath):
                        with open(tpath, "r", encoding="utf-8", errors="ignore") as fp:
                            transcript = fp.read()
                if self.wav_dir and ap:
                    base = os.path.splitext(os.path.basename(ap))[0]
                    wp = os.path.join(self.wav_dir, base + ".wav")
                    if not os.path.isfile(wp):
                        wp = os.path.join(self.wav_dir, base + ".WAV")
                    duration = _wav_duration_sec(wp) if os.path.isfile(wp) else 1.0
                hand_rows.append(compute_hand_feats_row(transcript, duration))
            features["hand"] = np.stack(hand_rows, axis=0)
        else:
            features["hand"] = np.zeros((nq, 4), dtype=np.float32)

        features = _apply_feat_norm_numpy(features, self._feat_norm)

        if self.classification:
            if len(self.label_col) != 1:
                raise ValueError("classification 模式下 label_col 只能包含一列，例如 g_level")
            col = self.label_col[0]
            sid = str(sample_id)
            if self.glevel_dict is not None:
                if sid not in self.glevel_dict:
                    raise KeyError(f"id {sid} 不在 glevel_csv 中")
                raw_y = self.glevel_dict[sid]
            else:
                raw_y = self.result_dict[sample_id][col]
            y = encode_g_level(raw_y, int_encoding=self._glevel_int_enc)
            label = torch.tensor(y, dtype=torch.long)
        else:
            label_normalized = np.array([(self.result_dict[sample_id][col] - 1) / 4 for col in self.label_col])
            label = torch.tensor(label_normalized, dtype=torch.float32)

        return (
            {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()},
            label,
            str(sample_id),
        )

class MultimodalDatasetForTestT2(Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        video_dir,
        text_dir,
        question,
        rating_csv,
        args=None,
        fallback_audio_dir=None,
        fallback_video_dir=None,
        fallback_text_dir=None,
        fallback_val_audio_dir=None,
        fallback_val_video_dir=None,
        fallback_val_text_dir=None,
    ):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.fallback_audio_dir = fallback_audio_dir
        self.fallback_video_dir = fallback_video_dir
        self.fallback_text_dir = fallback_text_dir
        self.fallback_val_audio_dir = fallback_val_audio_dir
        self.fallback_val_video_dir = fallback_val_video_dir
        self.fallback_val_text_dir = fallback_val_text_dir
        if isinstance(question, list) and len(question) == 1 and isinstance(question[0], str):
            self.question = ast.literal_eval(question[0])
        else:
            self.question = question
        if args and getattr(args, "modalities", None):
            self.training_modal = frozenset(str(m).strip() for m in args.modalities)
        else:
            self.training_modal = frozenset({"audio", "video", "text"})
        self.audio_dim = int(getattr(args, "audio_dim", 512)) if args else 512
        self.video_dim = int(getattr(args, "video_dim", 512)) if args else 512
        self.text_dim = int(getattr(args, "text_dim", 768)) if args else 768
        self.use_hand = not (args and getattr(args, "no_hand", False))
        self._feat_norm = _load_feat_norm_bundle(args)
        self.rating = pd.read_csv(rating_csv)

        self.result_dict = {}
        for _, row in self.rating.iterrows():
            key = row["id"]
            self.result_dict[key] = row
        self.transcript_dir = getattr(args, "transcript_dir", None) if args else None
        self.wav_dir = getattr(args, "wav_dir", None) if args else None
        if self.transcript_dir == "":
            self.transcript_dir = None
        if self.wav_dir == "":
            self.wav_dir = None

        self._npy_audio = _list_npy_filenames(self.audio_dir)
        self._npy_audio_fb = _list_npy_filenames(self.fallback_audio_dir)
        self._npy_audio_fv = _list_npy_filenames(self.fallback_val_audio_dir)
        self._npy_video = _list_npy_filenames(self.video_dir)
        self._npy_video_fb = _list_npy_filenames(self.fallback_video_dir)
        self._npy_video_fv = _list_npy_filenames(self.fallback_val_video_dir)
        self._npy_text = _list_npy_filenames(self.text_dir)
        self._npy_text_fb = _list_npy_filenames(self.fallback_text_dir)
        self._npy_text_fv = _list_npy_filenames(self.fallback_val_text_dir)
        _nodrop = args is not None and getattr(args, "no_drop_incomplete_features", False)
        if not _nodrop:
            self.data = _drop_rows_missing_features(
                self.data,
                self.question,
                self.audio_dir,
                self.video_dir,
                self.text_dir,
                self.fallback_audio_dir,
                self.fallback_video_dir,
                self.fallback_text_dir,
                split_name=f"test:{os.path.basename(csv_file)}",
                la=self._npy_audio,
                lab=self._npy_audio_fb,
                lv=self._npy_video,
                lvb=self._npy_video_fb,
                lt=self._npy_text,
                ltb=self._npy_text_fb,
                allow_empty=True,
                modalities_required=self.training_modal,
                fallback_audio_dir_2=self.fallback_val_audio_dir,
                lab2=self._npy_audio_fv,
                fallback_video_dir_2=self.fallback_val_video_dir,
                lvb2=self._npy_video_fv,
                fallback_text_dir_2=self.fallback_val_text_dir,
                ltb2=self._npy_text_fv,
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise IndexError("empty test dataset")
        sample_id = self.data.iloc[idx]['id']
        audio_paths, video_paths, text_paths = [], [], []
        features = {}

        for q in self.question:
            ap = vp = tp = None
            if "audio" in self.training_modal:
                ap = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.audio_dir,
                    self.fallback_audio_dir,
                    self._npy_audio,
                    self._npy_audio_fb,
                    self.fallback_val_audio_dir,
                    self._npy_audio_fv,
                )
            if "video" in self.training_modal:
                vp = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.video_dir,
                    self.fallback_video_dir,
                    self._npy_video,
                    self._npy_video_fb,
                    self.fallback_val_video_dir,
                    self._npy_video_fv,
                )
            if "text" in self.training_modal:
                tp = _resolve_from_name_lists(
                    sample_id,
                    q,
                    self.text_dir,
                    self.fallback_text_dir,
                    self._npy_text,
                    self._npy_text_fb,
                    self.fallback_val_text_dir,
                    self._npy_text_fv,
                )
            if "audio" in self.training_modal and ap is None:
                tried_a = self.audio_dir
                if self.fallback_audio_dir:
                    tried_a = f"{tried_a} 与 {self.fallback_audio_dir}"
                if self.fallback_val_audio_dir:
                    tried_a = f"{tried_a} 与 val:{self.fallback_val_audio_dir}"
                raise FileNotFoundError(
                    f"Missing audio for {sample_id}_{q}（已查找: {tried_a}）。"
                    "若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            if "video" in self.training_modal and vp is None:
                raise FileNotFoundError(
                    f"Missing video for {sample_id}_{q}。若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            if "text" in self.training_modal and tp is None:
                raise FileNotFoundError(
                    f"Missing text for {sample_id}_{q}。若刚更新过特征目录，请重启训练进程以刷新缓存。"
                )
            audio_paths.append(ap)
            video_paths.append(vp)
            text_paths.append(tp)

        nq = len(self.question)
        if "audio" in self.training_modal:
            features["audio"] = np.concatenate(
                [np.expand_dims(np.load(p), axis=0) for p in audio_paths], axis=0
            )
        else:
            features["audio"] = np.zeros((nq, self.audio_dim), dtype=np.float32)

        if "video" in self.training_modal:
            features["video"] = np.stack([np.load(p) for p in video_paths], axis=0)
        else:
            features["video"] = np.zeros((nq, self.video_dim), dtype=np.float32)

        if "text" in self.training_modal:
            features["text"] = np.stack([np.load(p) for p in text_paths], axis=0)
        else:
            features["text"] = np.zeros((nq, self.text_dim), dtype=np.float32)

        if self.use_hand:
            hand_rows = []
            for ap, vp, tp in zip(audio_paths, video_paths, text_paths):
                ref = tp or ap or vp
                if ref is None:
                    hand_rows.append(compute_hand_feats_row("", 1.0))
                    continue
                af = os.path.basename(ap) if ap else os.path.basename(ref)
                tf = os.path.basename(tp) if tp else os.path.basename(ref)
                transcript = ""
                duration = 1.0
                if self.transcript_dir:
                    base = os.path.splitext(tf)[0]
                    tpath = os.path.join(self.transcript_dir, base + ".txt")
                    if os.path.isfile(tpath):
                        with open(tpath, "r", encoding="utf-8", errors="ignore") as fp:
                            transcript = fp.read()
                if self.wav_dir and ap:
                    base = os.path.splitext(os.path.basename(ap))[0]
                    wp = os.path.join(self.wav_dir, base + ".wav")
                    if not os.path.isfile(wp):
                        wp = os.path.join(self.wav_dir, base + ".WAV")
                    duration = _wav_duration_sec(wp) if os.path.isfile(wp) else 1.0
                hand_rows.append(compute_hand_feats_row(transcript, duration))
            features["hand"] = np.stack(hand_rows, axis=0)
        else:
            features["hand"] = np.zeros((nq, 4), dtype=np.float32)

        features = _apply_feat_norm_numpy(features, self._feat_norm)
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}, sample_id


def collate_fn_train(batch):
    features_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]

    features = {}
    masks = {}
    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            modality_tensors = [item[0][k] for item in batch]
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, labels, sample_ids


def collate_fn_test(batch):
    # batch: List of (features_dict, sample_id)
    features_list = [item[0] for item in batch]
    sample_ids = [item[1] for item in batch]

    features = {}
    masks = {}

    for k in features_list[0].keys():
        modality_tensors = [f[k] for f in features_list]

        if k == 'audio':
            lengths = [t.shape[0] for t in modality_tensors]
            max_len = max(lengths)
            padded = pad_sequence(modality_tensors, batch_first=True)  # (B, T, D)
            mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
            audio_mask = mask.float()
            features[k] = padded
            masks[k + "_mask"] = audio_mask
        else:
            features[k] = torch.stack(modality_tensors)

    return features, masks, sample_ids