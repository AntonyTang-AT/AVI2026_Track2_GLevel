# 多模态 g_level：数据路径默认值（与 vote_train_glevel.sh / vote_test_glevel.sh 一致）
# 用法：须先 cd 到工程根（PWD=项目根）再 source；可通过 export 覆盖任意变量后再 source。
# Nanbeige：工程根下 data/text_nb、data/text_nb_val、data/test_nb 若存在则优先（避免 FEAT_*/text_nb 指向未挂载的 Super-Lu）。
# 可改工程根： export GLEVEL_REPO_ROOT=/path/to/AVI2026_Track2_GLevel

_repo_root="${GLEVEL_REPO_ROOT:-${PWD}}"
SUP_DATASET="${SUPERLU_DATASET:-/data/Super-Lu/dataset}"

if [ -f "${_repo_root}/data/train_data.csv" ]; then
  TRAIN_CSV="${TRAIN_CSV:-${_repo_root}/data/train_data.csv}"
else
  TRAIN_CSV="${TRAIN_CSV:-${SUP_DATASET}/train_data.csv}"
fi
if [ -f "${_repo_root}/data/val_data.csv" ]; then
  VAL_CSV="${VAL_CSV:-${_repo_root}/data/val_data.csv}"
else
  VAL_CSV="${VAL_CSV:-${SUP_DATASET}/val_data.csv}"
fi
if [ -f "${_repo_root}/data/test_data_basic_information.csv" ]; then
  TEST_CSV="${TEST_CSV:-${_repo_root}/data/test_data_basic_information.csv}"
else
  TEST_CSV="${TEST_CSV:-${SUP_DATASET}/test_data_basic_information.csv}"
fi
if [ -f "${_repo_root}/data/train_data.csv" ]; then
  RATING_CSV="${RATING_CSV:-${_repo_root}/data/train_data.csv}"
else
  RATING_CSV="${RATING_CSV:-${SUP_DATASET}/train_data.csv}"
fi

FEAT_TRAIN="${FEAT_TRAIN:-${SUP_DATASET}/train_feature}"
FEAT_VAL="${FEAT_VAL:-${SUP_DATASET}/val_feature}"
# 若工程内已复制官方测试集 audio/video（如 rsync 自 Super-Lu），默认用本地路径便于离线提交
if [ -d "${_repo_root}/data/test_feature/video" ] && [ -d "${_repo_root}/data/test_feature/audio" ]; then
  FEAT_TEST="${FEAT_TEST:-${_repo_root}/data/test_feature}"
else
  FEAT_TEST="${FEAT_TEST:-${SUP_DATASET}/test_feature}"
fi

_NB_SUB="${NANBEIGE_TEXT_SUBDIR:-text_nb}"
if [ "${NANBEIGE_TEXT:-0}" = "1" ]; then
  TEXT_DIM="${TEXT_DIM:-2560}"
  _local_nb_train="${_repo_root}/data/text_nb"
  _local_nb_val="${_repo_root}/data/text_nb_val"
  if [ -d "${_local_nb_train}" ]; then
    TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${_local_nb_train}}"
  else
    TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${FEAT_TRAIN}/${_NB_SUB}}"
  fi
  case "${_NB_SUB}" in
    *smoke*)
      TEXT_VAL_DIR="${TEXT_VAL_DIR:-${TEXT_TRAIN_DIR}}"
      TEXT_TEST_DIR="${TEXT_TEST_DIR:-${TEXT_TRAIN_DIR}}"
      ;;
    *)
      if [ -d "${_local_nb_val}" ]; then
        TEXT_VAL_DIR="${TEXT_VAL_DIR:-${_local_nb_val}}"
      else
        TEXT_VAL_DIR="${TEXT_VAL_DIR:-${FEAT_VAL}/${_NB_SUB}}"
      fi
      _repo_test_nb="${_repo_root}/data/test_nb"
      if [ -d "${_repo_test_nb}" ]; then
        TEXT_TEST_DIR="${TEXT_TEST_DIR:-${_repo_test_nb}}"
      else
        TEXT_TEST_DIR="${TEXT_TEST_DIR:-${FEAT_TEST}/${_NB_SUB}}"
      fi
      ;;
  esac
else
  TEXT_DIM="${TEXT_DIM:-768}"
  TEXT_TRAIN_DIR="${TEXT_TRAIN_DIR:-${FEAT_TRAIN}/text}"
  TEXT_VAL_DIR="${TEXT_VAL_DIR:-${FEAT_VAL}/text}"
  TEXT_TEST_DIR="${TEXT_TEST_DIR:-${FEAT_TEST}/text}"
fi
