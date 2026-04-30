#!/bin/sh

# Usage: ./download_datasets.sh [--purge] [output_dir]
#   --purge      Remove all existing datasets before downloading
#   output_dir   Directory to download datasets into (default: ./raw_datasets)

PURGE=false
OUTPUT_DIR=""

for arg in "$@"; do
    if [ "$arg" = "--purge" ]; then
        PURGE=true
    else
        OUTPUT_DIR="$arg"
    fi
done

OUTPUT_DIR="${OUTPUT_DIR:-./raw_datasets}"

if [ "$PURGE" = true ]; then
    echo "Purging all datasets in ${OUTPUT_DIR}..."
    rm -rf "$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Note: The blog_authorship_corpus download is currently disabled due to an unreliable server.
# curl -k -O http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip

if [ ! -d "hate-speech-dataset" ]; then
    echo "Downloading hate-speech-dataset..."
    git clone --depth 1 https://github.com/Vicomtech/hate-speech-dataset.git
    rm -rf hate-speech-dataset/.git
else
    echo "Skipping hate-speech-dataset (already exists)"
fi

if [ ! -d "hate-speech-and-offensive-language" ]; then
    echo "Downloading hate-speech-and-offensive-language..."
    git clone --depth 1 https://github.com/t-davidson/hate-speech-and-offensive-language.git
    rm -rf hate-speech-and-offensive-language/.git
else
    echo "Skipping hate-speech-and-offensive-language (already exists)"
fi

if [ ! -d "CharCnn_Keras" ]; then
    echo "Downloading CharCnn_Keras..."
    git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras.git
    rm -rf CharCnn_Keras/.git
else
    echo "Skipping CharCnn_Keras (already exists)"
fi

if [ ! -d "enron_authorship_corpus" ]; then
    echo "Downloading enron_authorship_corpus..."
    curl https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/f0527105-2774-423e-80ca-cc692b70b6cb --output enron.zip
    unzip enron.zip
    mv 'Enron (80 authors)' enron_authorship_corpus
else
    echo "Skipping enron_authorship_corpus (already exists)"
fi

if [ ! -d "jigsaw-toxic-comment-classification-challenge" ]; then
    echo "Downloading jigsaw-toxic-comment-classification-challenge..."
    gdown https://drive.google.com/file/d/17eLjiDLhXXtrfqC1FzToHKrwTcwsAolN/view?usp=sharing --fuzzy
    tar zxvf jigsaw-toxic-comment-classification-challenge.tar.gz
    rm jigsaw-toxic-comment-classification-challenge.tar.gz
else
    echo "Skipping jigsaw-toxic-comment-classification-challenge (already exists)"
fi

if [ ! -d "dummy_retrieval" ]; then
    echo "Downloading dummy_retrieval..."
    mkdir ./dummy_retrieval
    cd ./dummy_retrieval
    gdown https://drive.google.com/file/d/17HkJs9E5nnwuDfONiUy3sw0-Nj53SwvL/view?usp=sharing --fuzzy
    cd ..
else
    echo "Skipping dummy_retrieval (already exists)"
fi

if [ ! -d "fanfiction_retrieval" ]; then
    echo "Downloading fanfiction_retrieval..."
    mkdir fanfiction_retrieval
    cd fanfiction_retrieval
    gdown https://drive.google.com/file/d/1imzUA2kg4782WnYWxlKICyk1MmPCCtsV/view?usp=sharing --fuzzy
    tar zxvf fanfiction_retrieval.tar.gz
    rm fanfiction_retrieval.tar.gz
    cd ..
else
    echo "Skipping fanfiction_retrieval (already exists)"
fi

if [ ! -d "amazon_retrieval" ]; then
    echo "Downloading amazon_retrieval..."
    mkdir amazon_retrieval
    cd amazon_retrieval
    gdown https://drive.google.com/file/d/1HFWoFk3V7vqM_DbZzd8ixecdbjHp29c4/view?usp=sharing --fuzzy
    tar zxvf amazon_retrieval.tar.gz
    rm amazon_retrieval.tar.gz
    cd ..
else
    echo "Skipping amazon_retrieval (already exists)"
fi

if [ ! -d "stackexchange_retrieval" ]; then
    echo "Downloading stackexchange_retrieval..."
    mkdir stackexchange_retrieval
    cd stackexchange_retrieval
    gdown https://drive.google.com/file/d/1PVu0BI6rvo9MndcIdHOhRdqm4nfaqgSG/view?usp=drive_link --fuzzy
    tar zxvf stackexchange_retrieval.tar.gz
    rm stackexchange_retrieval.tar.gz
    cd ..
else
    echo "Skipping stackexchange_retrieval (already exists)"
fi

# Graded Formality GPT-5-mini
if [ ! -d "graded_formality" ]; then
    echo "Downloading graded_formality..."
    mkdir -p graded_formality
    cd graded_formality
    gdown https://drive.google.com/file/d/1TKtgg-6j2Yd-GTfiq0nnHRPcZC-VpGnu/view?usp=drive_link --fuzzy
    unzip graded_formality_generated_gpt-5-mini.zip
    rm graded_formality_generated_gpt-5-mini.zip
    rm -rf __MACOSX
    cd ..
else
    echo "Skipping graded_formality (already exists)"
fi

if [ ! -d "STEL" ]; then
    echo "Downloading STEL..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/nlpsoc/STEL.git STEL
    cd STEL
    git sparse-checkout set Data/STEL
    rm -rf .git
    cd ..
else
    echo "Skipping STEL (already exists)"
fi

# OneStopEnglishCorpus (Texts-SeparatedByReadingLevel)
if [ ! -d "OneStopEnglishCorpus" ]; then
    echo "Downloading OneStopEnglishCorpus..."
    git clone --depth 1 https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git OneStopEnglishCorpus
    rm -rf OneStopEnglishCorpus/.git
else
    echo "Skipping OneStopEnglishCorpus (already exists)"
fi

# PAN AV-15
if [ ! -d "pan15-authorship-verification-test-dataset2-2015-04-19" ]; then
    echo "Downloading PAN AV-15..."
    wget https://zenodo.org/records/3737563/files/pan15-authorship-verification-test-and-training.zip
    unzip pan15-authorship-verification-test-and-training.zip
    unzip pan15-authorship-verification-test-dataset2-2015-04-19.zip
    rm pan15-authorship-verification-test-and-training.zip
    rm pan15-authorship-verification-test-dataset2-2015-04-19.zip
    rm pan15-authorship-verification-training-dataset-2015-04-19.zip
    cd pan15-authorship-verification-test-dataset2-2015-04-19
    unzip "*.zip"
    rm *.zip
    cd ../
else
    echo "Skipping PAN AV-15 (already exists)"
fi

# PAN AV-14
if [ ! -d "pan14-authorship-verification-test-2014-04-22" ]; then
    echo "Downloading PAN AV-14..."
    mkdir pan14-authorship-verification-test-2014-04-22
    cd pan14-authorship-verification-test-2014-04-22
    wget https://zenodo.org/records/3716033/files/pan14-authorship-verification-test-and-training.zip
    unzip pan14-authorship-verification-test-and-training.zip
    unzip pan14-authorship-verification-test-corpus1-2014-04-22.zip
    unzip pan14-authorship-verification-test-corpus2-2014-04-22.zip
    unzip -o "*test-corpus1*.zip"
    unzip -o "*test-corpus2*.zip"
    rm *.zip
    cd ../
else
    echo "Skipping PAN AV-14 (already exists)"
fi

# PAN AV-13
if [ ! -d "pan13-authorship-verification-test-corpus2-2013-05-29" ]; then
    echo "Downloading PAN AV-13..."
    wget https://zenodo.org/records/3715999/files/pan13-authorship-verification-test-and-training.zip
    unzip pan13-authorship-verification-test-and-training.zip
    unzip pan13-authorship-verification-test-corpus2-2013-05-29.zip
    rm *.zip
else
    echo "Skipping PAN AV-13 (already exists)"
fi

# PAN AV-20
if [ ! -d "pan20-authorship-verification-test" ]; then
    echo "Downloading PAN AV-20..."
    wget https://zenodo.org/records/5106099/files/pan20-authorship-verification-test.zip
    unzip pan20-authorship-verification-test.zip
    rm pan20-authorship-verification-test.zip
    echo "Subsampling PAN AV-20 to 500 pairs..."
    python3 -c "
import json, random

random.seed(42)
data_dir = 'pan20-authorship-verification-test'

with open(f'{data_dir}/pan20-authorship-verification-test.jsonl') as f:
    data = [json.loads(line) for line in f]
with open(f'{data_dir}/pan20-authorship-verification-test-truth.jsonl') as f:
    truths = [json.loads(line) for line in f]

truth_by_id = {t['id']: t for t in truths}

same = [d for d in data if truth_by_id[d['id']]['same']]
diff = [d for d in data if not truth_by_id[d['id']]['same']]

random.shuffle(same)
random.shuffle(diff)
selected = same[:250] + diff[:250]

selected_ids = {d['id'] for d in selected}

with open(f'{data_dir}/pan20-authorship-verification-test.jsonl', 'w') as f:
    for d in selected:
        f.write(json.dumps(d) + '\n')
with open(f'{data_dir}/pan20-authorship-verification-test-truth.jsonl', 'w') as f:
    for t in truths:
        if t['id'] in selected_ids:
            f.write(json.dumps(t) + '\n')

print(f'  Kept {len(selected)} pairs (250 same, 250 different)')
"
else
    echo "Skipping PAN AV-20 (already exists)"
fi

# PAN AV-21
if [ ! -d "pan21-authorship-verification-test" ]; then
    echo "Downloading PAN AV-21..."
    wget https://zenodo.org/records/5106099/files/pan21-authorship-verification-test.zip
    unzip pan21-authorship-verification-test.zip
    rm pan21-authorship-verification-test.zip
    echo "Subsampling PAN AV-21 to 500 pairs..."
    python3 -c "
import json, random

random.seed(42)
data_dir = 'pan21-authorship-verification-test'

with open(f'{data_dir}/pan21-authorship-verification-test.jsonl') as f:
    data = [json.loads(line) for line in f]
with open(f'{data_dir}/pan21-authorship-verification-test-truth.jsonl') as f:
    truths = [json.loads(line) for line in f]

truth_by_id = {t['id']: t for t in truths}

same = [d for d in data if truth_by_id[d['id']]['same']]
diff = [d for d in data if not truth_by_id[d['id']]['same']]

random.shuffle(same)
random.shuffle(diff)
selected = same[:250] + diff[:250]

selected_ids = {d['id'] for d in selected}

with open(f'{data_dir}/pan21-authorship-verification-test.jsonl', 'w') as f:
    for d in selected:
        f.write(json.dumps(d) + '\n')
with open(f'{data_dir}/pan21-authorship-verification-test-truth.jsonl', 'w') as f:
    for t in truths:
        if t['id'] in selected_ids:
            f.write(json.dumps(t) + '\n')

print(f'  Kept {len(selected)} pairs (250 same, 250 different)')
"
else
    echo "Skipping PAN AV-21 (already exists)"
fi

# GEDE Essay Detection
if [ ! -d "gede_essay_detection" ]; then
    echo "Downloading gede_essay_detection..."
    mkdir -p gede_essay_detection
    gdown https://drive.google.com/file/d/1c3x_CR44ZCUqHf1dHVPm7K04ZIbTSYoD/view?usp=drive_link --fuzzy
    tar -zxvf gede_essay_detection.tar.gz
    rm gede_essay_detection.tar.gz
else
    echo "Skipping gede_essay_detection (already exists)"
fi

# CORE Register Corpus
if [ ! -d "CORE-corpus" ]; then
    echo "Downloading CORE-corpus..."
    mkdir -p CORE-corpus
    cd CORE-corpus
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/train.tsv.gz
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/dev.tsv.gz
    wget https://github.com/TurkuNLP/CORE-corpus/raw/master/test.tsv.gz
    gunzip train.tsv.gz dev.tsv.gz test.tsv.gz
    cd ..
else
    echo "Skipping CORE-corpus (already exists)"
fi

# RadioTalk
if [ ! -d "radiotalk" ]; then
    echo "Downloading radiotalk..."
    mkdir -p radiotalk
    cd radiotalk
    gdown https://drive.google.com/file/d/1NHgHZDlMB9Yh9SGQdSTjz_CNLZZqGCuW/view?usp=drive_link --fuzzy
    tar -zxvf radiotalk.tar.gz
    rm radiotalk.tar.gz
    cd ..
else
    echo "Skipping radiotalk (already exists)"
fi

# Machine Text Detection - M4

if [ ! -d "M4" ]; then
    echo "Downloading M4..."
    git clone https://github.com/mbzuai-nlp/M4.git
    rm -rf M4/.git
else
    echo "Skipping M4 (already exists)"
fi

if [ ! -d "DetectRL" ]; then
    echo "Downloading DetectRL..."
    git clone https://github.com/NLP2CT/DetectRL.git
    rm -rf DetectRL/.git
else
    echo "Skipping DetectRL (already exists)"
fi

# PAN24 Generative Authorship (news)
if [ ! -d "pan24-generative-authorship-news" ]; then
    echo "Downloading PAN24 Generative Authorship (news)..."
    curl -L -o pan24-generative-authorship-news.zip https://zenodo.org/records/10718757/files/pan24-generative-authorship-news.zip
    unzip -q pan24-generative-authorship-news.zip
    rm pan24-generative-authorship-news.zip
else
    echo "Skipping pan24-generative-authorship-news (already exists)"
fi

# PAN18 Style Change Detection — validation set only
if [ ! -d "pan18-style-change" ]; then
    echo "Downloading PAN18 Style Change Detection..."
    curl -L -o pan18-style-change.zip https://zenodo.org/records/3746040/files/pan18-style-change-detection-validation-dataset-2018-01-31.zip
    mkdir -p pan18-style-change/_tmp
    unzip -q pan18-style-change.zip -d pan18-style-change/_tmp
    mv pan18-style-change/_tmp/pan18-style-change-detection-validation-dataset-2018-01-31/* pan18-style-change/
    rm -rf pan18-style-change/_tmp pan18-style-change.zip
else
    echo "Skipping pan18-style-change (already exists)"
fi

# PAN22 Style Change Detection — three tasks, validation sets only
if [ ! -d "pan22-style-change" ]; then
    echo "Downloading PAN22 Style Change Detection..."
    curl -L -o pan22.zip https://zenodo.org/records/6334245/files/pan22.zip
    mkdir -p pan22-style-change/_tmp
    unzip -q pan22.zip -d pan22-style-change/_tmp
    for i in 1 2 3; do
        case $i in
            1) task=basic ;;
            2) task=advanced ;;
            3) task=sentence ;;
        esac
        mkdir -p pan22-style-change/$task
        mv pan22-style-change/_tmp/dataset$i/validation/* pan22-style-change/$task/
    done
    rm -rf pan22-style-change/_tmp pan22.zip
else
    echo "Skipping pan22-style-change (already exists)"
fi

# PAN23 Style Change Detection (multi-author writing style analysis) — validation sets only
if [ ! -d "pan23-style-change" ]; then
    echo "Downloading PAN23 Style Change Detection..."
    curl -L -o pan23-multi-author-analysis.zip https://zenodo.org/records/7729178/files/pan23-multi-author-analysis.zip
    mkdir -p pan23-style-change/_tmp
    unzip -q pan23-multi-author-analysis.zip -d pan23-style-change/_tmp
    pan23_base="pan23-style-change/_tmp/release"
    for i in 1 2 3; do
        case $i in
            1) diff=easy ;;
            2) diff=medium ;;
            3) diff=hard ;;
        esac
        mkdir -p pan23-style-change/$diff
        mv "$pan23_base/pan23-multi-author-analysis-dataset$i/pan23-multi-author-analysis-dataset$i-validation"/* pan23-style-change/$diff/
    done
    rm -rf pan23-style-change/_tmp pan23-multi-author-analysis.zip
else
    echo "Skipping pan23-style-change (already exists)"
fi

# PAN24 Style Change Detection (multi-author writing style analysis) — validation sets only
if [ ! -d "pan24-style-change" ]; then
    echo "Downloading PAN24 Style Change Detection..."
    curl -L -o pan24-multi-author-analysis.zip https://zenodo.org/records/10677876/files/pan24-multi-author-analysis.zip
    mkdir -p pan24-style-change/_tmp
    unzip -q pan24-multi-author-analysis.zip -d pan24-style-change/_tmp
    for diff in easy medium hard; do
        mkdir -p pan24-style-change/$diff
        mv pan24-style-change/_tmp/$diff/validation/* pan24-style-change/$diff/
    done
    rm -rf pan24-style-change/_tmp pan24-multi-author-analysis.zip
else
    echo "Skipping pan24-style-change (already exists)"
fi

# PAN25 Style Change Detection (multi-author writing style analysis) — validation sets only
if [ ! -d "pan25-style-change" ]; then
    echo "Downloading PAN25 Style Change Detection..."
    curl -L -o pan25-multi-author-analysis.zip https://zenodo.org/records/14891299/files/pan25-multi-author-analysis.zip
    mkdir -p pan25-style-change/_tmp
    unzip -q pan25-multi-author-analysis.zip -d pan25-style-change/_tmp
    for diff in easy medium hard; do
        mkdir -p pan25-style-change/$diff
        mv pan25-style-change/_tmp/$diff/validation/* pan25-style-change/$diff/
    done
    rm -rf pan25-style-change/_tmp pan25-multi-author-analysis.zip
else
    echo "Skipping pan25-style-change (already exists)"
fi

# PAN26 Style Change Detection (multi-author writing style analysis) — validation sets only
if [ ! -d "pan26-style-change" ]; then
    echo "Downloading PAN26 Style Change Detection..."
    curl -L -o mawsa26-pan-zenodo.zip https://zenodo.org/records/19068843/files/mawsa26-pan-zenodo.zip
    unzip -q mawsa26-pan-zenodo.zip
    mkdir -p pan26-style-change
    for diff in easy medium hard; do
        mkdir -p pan26-style-change/$diff
        mv mawsa26-pan-zenodo/$diff/validation/* pan26-style-change/$diff/
    done
    rm -rf mawsa26-pan-zenodo mawsa26-pan-zenodo.zip
else
    echo "Skipping pan26-style-change (already exists)"
fi

# PAN25 Generative AI Detection (Task 2) — dev set only
if [ ! -d "pan25-generative-ai-detection-task2" ]; then
    echo "Downloading PAN25 Generative AI Detection (Task 2)..."
    curl -L -o pan25-generative-ai-detection-task2.zip https://zenodo.org/records/14966981/files/pan25-generative-ai-detection-task2-train.zip
    mkdir -p pan25-generative-ai-detection-task2
    unzip -q -j pan25-generative-ai-detection-task2.zip dev.jsonl -d pan25-generative-ai-detection-task2
    rm pan25-generative-ai-detection-task2.zip
else
    echo "Skipping pan25-generative-ai-detection-task2 (already exists)"
fi

# PAN25/26 Generative AI Detection (Task 1) — validation set only
# (Same data is used in both the 2025 and 2026 editions of the shared task.)
if [ ! -d "pan25-26-generative-ai-detection-task1" ]; then
    echo "Downloading PAN25/26 Generative AI Detection (Task 1)..."
    curl -L -o pan25-26-generative-ai-detection-task1.zip https://zenodo.org/records/14962653/files/pan25-generative-ai-detection-task1-train.zip
    mkdir -p pan25-26-generative-ai-detection-task1
    unzip -q -j pan25-26-generative-ai-detection-task1.zip val.jsonl -d pan25-26-generative-ai-detection-task1
    rm pan25-26-generative-ai-detection-task1.zip
else
    echo "Skipping pan25-26-generative-ai-detection-task1 (already exists)"
fi

# PAN18 Cross-Domain Authorship Attribution
if [ ! -d "pan18_cross_domain_authorship_attribution" ]; then
    echo "Downloading PAN18 Cross-Domain Authorship Attribution..."
    wget https://zenodo.org/records/3737849/files/pan18-cross-domain-authorship-attribution-dataset.zip
    unzip pan18-cross-domain-authorship-attribution-dataset.zip
    unzip pan18-cross-domain-authorship-attribution-test-dataset2-2018-04-20.zip
    rm pan18-cross-domain-authorship-attribution-dataset.zip
    rm pan18-cross-domain-authorship-attribution-test-dataset2-2018-04-20.zip
    rm pan18-cross-domain-authorship-attribution-training-dataset-2017-12-02.zip

    echo "Converting PAN18 to retrieval JSONL format..."
    python3 -c "
import json, os

test_dir = 'pan18-cross-domain-authorship-attribution-test-dataset2-2018-04-20'
out_dir = 'pan18_cross_domain_authorship_attribution'
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(test_dir, 'collection-info.json')) as f:
    collection = json.load(f)

lang_problems = {}
for entry in collection:
    lang = entry['language']
    lang_problems.setdefault(lang, []).append(entry['problem-name'])

lang_names = {'en': 'english', 'fr': 'french', 'it': 'italian', 'pl': 'polish', 'sp': 'spanish'}

for lang, problems in sorted(lang_problems.items()):
    records = []
    for problem_name in problems:
        problem_dir = os.path.join(test_dir, problem_name)
        with open(os.path.join(problem_dir, 'ground-truth.json')) as f:
            gt = json.load(f)
        truth_map = {e['unknown-text']: e['true-author'] for e in gt['ground_truth']}

        # Candidate targets: concatenate known texts per author
        for candidate_dir in sorted(os.listdir(problem_dir)):
            candidate_path = os.path.join(problem_dir, candidate_dir)
            if not os.path.isdir(candidate_path) or candidate_dir == 'unknown':
                continue
            known_files = sorted(f for f in os.listdir(candidate_path) if f.endswith('.txt'))
            texts = []
            for kf in known_files:
                with open(os.path.join(candidate_path, kf), encoding='utf-8', errors='replace') as f:
                    texts.append(f.read())
            label = f'{problem_name}_{candidate_dir}'
            records.append({'text': '\n\n'.join(texts), 'label': label, 'is_query': False})

        # Unknown queries
        unknown_dir = os.path.join(problem_dir, 'unknown')
        for uf in sorted(os.listdir(unknown_dir)):
            if not uf.endswith('.txt'):
                continue
            with open(os.path.join(unknown_dir, uf), encoding='utf-8', errors='replace') as f:
                text = f.read()
            true_author = truth_map.get(uf)
            if true_author is None:
                continue
            label = f'{problem_name}_{true_author}'
            records.append({'text': text, 'label': label, 'is_query': True})

    lang_name = lang_names[lang]
    lang_dir = os.path.join(out_dir, lang_name)
    os.makedirs(lang_dir, exist_ok=True)
    out_file = os.path.join(lang_dir, f'pan18_{lang_name}.jsonl')
    with open(out_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    n_queries = sum(1 for r in records if r['is_query'])
    n_targets = sum(1 for r in records if not r['is_query'])
    print(f'  {lang_name}: {n_targets} targets, {n_queries} queries')
"
    rm -rf pan18-cross-domain-authorship-attribution-test-dataset2-2018-04-20
else
    echo "Skipping PAN18 Cross-Domain Authorship Attribution (already exists)"
fi

# Groenwold et al. 2020 AAVE/SAE parallel tweets (EMNLP 2020)
if [ ! -d "twitter_aave_sae" ]; then
    echo "Downloading twitter_aave_sae..."
    mkdir -p twitter_aave_sae
    curl -L -o twitter_aave_sae.zip https://aclanthology.org/attachments/2020.emnlp-main.473.OptionalSupplementaryMaterial.zip
    unzip -q -j twitter_aave_sae.zip "EMNLP-AAVE-files/aave_samples.txt" "EMNLP-AAVE-files/sae_samples.txt" -d twitter_aave_sae
    rm twitter_aave_sae.zip
else
    echo "Skipping twitter_aave_sae (already exists)"
fi

# Xu et al. 2012 parallel Shakespeare (COLING 2012)
if [ ! -d "parallel_shakespeare" ]; then
    echo "Downloading parallel_shakespeare..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/cocoxu/Shakespeare.git parallel_shakespeare_tmp
    git -C parallel_shakespeare_tmp sparse-checkout set data/align/plays/merged
    mkdir -p parallel_shakespeare
    cp parallel_shakespeare_tmp/data/align/plays/merged/*.snt.aligned parallel_shakespeare/
    rm -rf parallel_shakespeare_tmp
else
    echo "Skipping parallel_shakespeare (already exists)"
fi

# MASC 3.0.0 (Manually Annotated Sub-Corpus) — text genre source
if [ ! -d "MASC-3.0.0" ]; then
    echo "Downloading MASC-3.0.0..."
    wget --no-check-certificate https://www.anc.org/MASC/download/MASC-3.0.0.zip
    unzip -q MASC-3.0.0.zip
    rm MASC-3.0.0.zip
else
    echo "Skipping MASC-3.0.0 (already exists)"
fi

# Stanford politeness corpora (Wikipedia + Stack Exchange).
# Zip URLs taken from convokit's official download config:
#   https://github.com/CornellNLP/ConvoKit/blob/master/download_config.json
# (Same URLs convokit.download() resolves at runtime; using curl avoids
#  pulling the full convokit package as a dependency.)
if [ ! -d "wikipedia-politeness-corpus" ]; then
    echo "Downloading wikipedia-politeness-corpus..."
    curl -L -o wikipedia-politeness-corpus.zip \
        https://zissou.infosci.cornell.edu/convokit/datasets/wikipedia-politeness-corpus/wikipedia-politeness-corpus.zip
    unzip -q wikipedia-politeness-corpus.zip
    rm wikipedia-politeness-corpus.zip
else
    echo "Skipping wikipedia-politeness-corpus (already exists)"
fi

if [ ! -d "stack-exchange-politeness-corpus" ]; then
    echo "Downloading stack-exchange-politeness-corpus..."
    curl -L -o stack-exchange-politeness-corpus.zip \
        https://zissou.infosci.cornell.edu/convokit/datasets/stack-exchange-politeness-corpus/stack-exchange-politeness-corpus.zip
    unzip -q stack-exchange-politeness-corpus.zip
    rm stack-exchange-politeness-corpus.zip
else
    echo "Skipping stack-exchange-politeness-corpus (already exists)"
fi

# FCE released dataset (Cambridge Learner Corpus, FCE subset). Used for L1
# (native-language) prediction. Only the dataset/ subfolder (the learner
# script XMLs) is extracted; outliers/, prompts/, README, license, and the
# examiner-scores file are skipped.
if [ ! -d "fce_l1" ]; then
    echo "Downloading fce_l1 (FCE released dataset)..."
    mkdir -p fce_l1
    curl -L -o fce_l1/fce-released-dataset.zip \
        https://s3-eu-west-1.amazonaws.com/ilexir-website-media/fce-released-dataset.zip
    unzip -q fce_l1/fce-released-dataset.zip 'fce-released-dataset/dataset/*' -d fce_l1
    rm fce_l1/fce-released-dataset.zip
else
    echo "Skipping fce_l1 (already exists)"
fi
    
# CEECES 1 + 2 (Corpus of Early English Correspondence Extension Samplers,
# University of Helsinki, VARIENG). 18th-century English letters labelled by
# 20-year period; used for historical period prediction (issue #94).
if [ ! -d "ceeces" ]; then
    echo "Downloading ceeces (CEECES 1 + CEECES 2)..."
    mkdir -p ceeces/CEECES1 ceeces/CEECES2
    curl -L -o ceeces/CEECES1/CEECES1-metadata.txt https://zenodo.org/records/6411789/files/CEECES1-metadata.txt
    curl -L -o ceeces/CEECES1/CEECES-1.zip         https://zenodo.org/records/6411789/files/CEECES-1.zip
    unzip -q ceeces/CEECES1/CEECES-1.zip -d ceeces/CEECES1
    rm ceeces/CEECES1/CEECES-1.zip
    curl -L -o ceeces/CEECES2/CEECES2-metadata.txt https://zenodo.org/records/5887101/files/CEECES2-metadata.txt
    curl -L -o ceeces/CEECES2/CEECES-2.zip         https://zenodo.org/records/5887101/files/CEECES-2.zip
    unzip -q ceeces/CEECES2/CEECES-2.zip -d ceeces/CEECES2
    rm ceeces/CEECES2/CEECES-2.zip
else
    echo "Skipping ceeces (already exists)"
fi

#### Probing

mkdir ./probing

if [ ! -f ./probing/blog_small.jsonl ]; then
    echo "Downloading blog_small.jsonl..."
    gdown https://drive.google.com/file/d/1V9GODxHA8L4f9DXzKC2f34doa4LfqqAH/view?usp=drive_link --fuzzy
    mv blog_small.jsonl ./probing/
else
    echo "Skipping blog_small.jsonl (already exists)"
fi

if [ ! -f ./probing/blog.jsonl ]; then
    echo "Downloading blog.jsonl..."
    gdown https://drive.google.com/file/d/1JU9F5SbPV8PaefcBwpy7wUCWrzNXsydO/view?usp=drive_link --fuzzy
    mv blog.jsonl ./probing/
else
    echo "Skipping blog.jsonl (already exists)"
fi

if [ ! -f ./probing/stackexchange.jsonl ]; then
    echo "Downloading stackexchange.jsonl..."
    gdown https://drive.google.com/file/d/1Ke_Re3kwfOmr2ljApcS8GP2CdsiNPHz7/view?usp=drive_link --fuzzy
    mv stackexchange.jsonl ./probing/
else
    echo "Skipping stackexchange.jsonl (already exists)"
fi

if [ ! -f ./probing/reddit.jsonl ]; then
    echo "Downloading reddit.jsonl..."
    gdown https://drive.google.com/file/d/1pU-Yo--OMtgG8qafk-7DBd_4WhYH_urc/view?usp=drive_link --fuzzy
    mv reddit.jsonl ./probing/
else
    echo "Skipping reddit.jsonl (already exists)"
fi

if [ ! -f ./probing/amazon.jsonl ]; then
    echo "Downloading amazon.jsonl..."
    gdown https://drive.google.com/file/d/1zIRqhSMEdQZ4EZq3-jCLpccuTvXsmCyx/view?usp=drive_link --fuzzy
    mv amazon.jsonl ./probing/
else
    echo "Skipping amazon.jsonl (already exists)"
fi

echo "Done downloading datasets."
