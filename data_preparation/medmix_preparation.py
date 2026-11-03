import os
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
import zipfile

import datasets

QA_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

Let's think step by step and include the final choice in the box \\boxed{{}}."""

CALC_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions.
QUESTION: {task_description}

When you are finished with all computations, please write your final answer value without any units, using the following formats in the boxed form \\boxed{{}}:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)"""

def build_question_with_options(question, options) -> str:
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key} {options[key]}")
    options_block = "; ".join(lines)
    return f"{question}\n{options_block}"


def build_question_with_note(question, note) -> str:
    return f"Patient Note:\n{note}\n\nQuestion:\n{question}"


def make_medqa_map_fn(data_source, split):
    def process_fn(example, idx):
        raw = example["data"]
        question = raw.get("Question")
        correct_option = raw.get("Correct Option")
        options = raw.get("Options")
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(example["id"]),
                "question": question_with_options,
            },
        }

    return process_fn


def make_headqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("qtext")
        answers = example.get("answers") # list of answer options
        ra = example.get("ra") # right answer id, e.g., "1", "2", ...
        category = example.get("category")
        qid = str(example.get("qid"))
        year = example.get("year")
        name = example.get("name")

        answers_sorted = sorted(answers, key=lambda a: int(a.get("aid", 0)))
        options = {}
        correct_letter = None
        # change answer format to options with letters
        for i, ans in enumerate(answers_sorted):
            letter = chr(ord("A") + i)
            options[letter] = ans.get("atext")
            if ra is not None and str(ans.get("aid")) == str(ra):
                correct_letter = letter

        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_letter},
            "extra_info": {
                "split": split,
                "qid": name + '_' + qid,
                "category": category,
                "year": year,
                "name": name,
                "question": question_with_options,
            },
        }

    return process_fn


def make_medmcqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options = {
            "A": example.get("opa"),
            "B": example.get("opb"),
            "C": example.get("opc"),
            "D": example.get("opd"),
        }
        cop = example.get("cop")  # correct option
        idx_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        correct_option = idx_map[int(cop)]

        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(example.get("id")),
                "subject_name": example.get("subject_name"),
                "topic_name": example.get("topic_name"),
                "choice_type": example.get("choice_type"),
                "explanation": example.get("exp"),
                "question": question_with_options,
            },
        }

    return process_fn


def make_medcalcbench_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("Question")
        note = example.get("Patient Note")
        answer = example.get("Ground Truth Answer")
        explanation = example.get("Ground Truth Explanation")

        question_with_note = build_question_with_note(question, note)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": calc_system_content},
                {"role": "user", "content": CALC_BASE_TEMPLATE.format(task_description=question_with_note)},
                # {"role": "user", "content": question_with_note},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "qid": str(example.get("Note ID")),
                "lower_bound": example.get("Lower Limit"),
                "upper_bound": example.get("Upper Limit"),
                "calculator_id": example.get("Calculator ID"),
                "calculator_name": example.get("Calculator Name"),
                "category": example.get("Category"),
                "output_type": example.get("Output Type"),
                "ground_truth_explanation": explanation,
                "row_number": example.get("Row Number"),
                "question": question_with_note,
            },
        }

    return process_fn


def make_medbullets_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options = {
            "A": example.get("choicesA"),
            "B": example.get("choicesB"),
            "C": example.get("choicesC"),
            "D": example.get("choicesD"),
            "E": example.get("choicesE"),
        }
        answer_idx = example.get("answer_idx")
        answer_text = example.get("answer")

        correct_option = str(answer_idx).strip().upper()
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "answer_text": answer_text,
                "explanation": example.get("explanation"),
                "question": question_with_options,
            },
        }

    return process_fn


def make_mmlu_pro_health_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        options_raw = example.get("options")
        answer = example.get("answer")
        answer_index = example.get("answer_index")
        category = example.get("category") or example.get("subject")
        question_id = example.get("question_id") or example.get("id")

        options = {chr(ord("A") + i): text for i, text in enumerate(options_raw)}
        correct_option = str(answer).strip().upper()
        question_with_options = build_question_with_options(question, options)

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question_with_options)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "answer_index": answer_index,
                "question_id": question_id,
                "category": category,
                "question": question_with_options,
            },
        }

    return process_fn


def make_medxpertqa_map_fn(data_source, split):
    def process_fn(example, idx):
        question = example.get("question")
        # options = example.get("options")
        correct_option = example.get("label")

        # question_with_options = f"{question}\n{user_content_suffix}"

        return {
            "data_source": data_source,
            "prompt": [
                # {"role": "system", "content": qa_system_content},
                {"role": "user", "content": QA_BASE_TEMPLATE.format(task_description=question)},
                # {"role": "user", "content": question_with_options},
            ],
            "ability": "medical",
            "reward_model": {"style": "rule", "ground_truth": correct_option},
            "extra_info": {
                "split": split,
                "qid": str(idx),
                "id": example.get("id"),
                "medical_task": example.get("medical_task"),
                "body_system": example.get("body_system"),
                "question_type": example.get("question_type"),
                "question": question,
            },
        }

    return process_fn


def filter_headqa_no_image(split):
    return split.filter(lambda x: x["image"] is None)


def normalize_question_text(text):
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def collect_ngrams_from_questions(dataset, n=16):
    ngrams = set()
    for example in dataset:
        question = example.get("extra_info", {}).get("question")
        question = normalize_question_text(question)
        if not question:
            continue
        tokens = question.split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return ngrams


def has_ngram_overlap(question, ngrams, n=16):
    question = normalize_question_text(question)
    if not question:
        return False
    tokens = question.split()
    if len(tokens) < n:
        return False
    for i in range(len(tokens) - n + 1):
        if " ".join(tokens[i : i + n]) in ngrams:
            return True
    return False


def cap_dataset_per_source(dataset, per_source_limit):
    kept_indices = []
    source_counts = defaultdict(int)
    for idx, example in enumerate(dataset):
        source = example.get("data_source")
        if source_counts[source] < per_source_limit:
            kept_indices.append(idx)
            source_counts[source] += 1
    return dataset.select(kept_indices), dict(source_counts)


# 固定官方入口；仅HeadQA经仓库内官方zip读取，不执行HF远程Python脚本。
DATASET_SPECS = [
    dict(name="openlifescienceai/medqa", revision="153e61cdd129eb79d3c27f82cdf3bc5e018c11b0", config=None,
         map_fn=make_medqa_map_fn, train_splits=["train", "dev"], test_splits=["test"]),
    dict(name="dvilares/head_qa", revision="d444ead98dd50d7949703c1662c7880a5c1ea77d", config="en",
         map_fn=make_headqa_map_fn, train_splits=["train", "validation"], test_splits=["test"], filter="no_image"),
    dict(name="openlifescienceai/medmcqa", revision="91c6572c454088bf71b679ad90aa8dffcd0d5868", config=None,
         map_fn=make_medmcqa_map_fn, train_splits=["train"], test_splits=["validation"]),
    dict(name="ncbi/MedCalc-Bench", revision="83a1a56d6b53f8d4c0e03f0ba9699becf524a595", config=None,
         map_fn=make_medcalcbench_map_fn, train_splits=["train"], test_splits=["test"]),
    dict(name="JesseLiu/medbulltes5op", revision="4e85cbaaf07a2435decd568e54b046fb955d8cd4", config=None,
         map_fn=make_medbullets_map_fn, train_splits=[], test_splits=["test"]),
    dict(name="TIGER-Lab/MMLU-Pro", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be", config=None,
         map_fn=make_mmlu_pro_health_map_fn, train_splits=[], test_splits=["test"], filter="health"),
    dict(name="TsinghuaC3I/MedXpertQA", revision="7e7c465a68eb2b866926bfa59c8c9d17a8daba65", config="Text",
         map_fn=make_medxpertqa_map_fn, train_splits=[], test_splits=["test"]),
]
MEDCALC_SOURCE = "ncbi/MedCalc-Bench"


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def validate_options(options):
    if not isinstance(options, dict) or not 2 <= len(options) <= 26:
        raise ValueError("options must contain 2..26 choices")
    if sorted(options) != [chr(65+i) for i in range(len(options))]:
        raise ValueError("options must use consecutive A..Z keys")
    for key, value in options.items():
        require_text(value, f"option {key}")


def validate_choice(label, options):
    validate_options(options)
    if not isinstance(label, str) or label not in options:
        raise ValueError(f"invalid label {label!r}")


def select_splits(dataset, split_names, source):
    missing = [name for name in split_names if name not in dataset]
    if missing:
        raise ValueError(f"{source}: required splits missing: {missing}")
    return [(name, dataset[name]) for name in split_names]


def validate_source_counts(train_counts, test_counts):
    expected_train = {s["name"] for s in DATASET_SPECS if s["train_splits"]}
    expected_test = {s["name"] for s in DATASET_SPECS}
    for kind, counts, expected in [("train", train_counts, expected_train), ("test", test_counts, expected_test)]:
        if set(counts) != expected or any(type(n) is not int or n <= 0 for n in counts.values()):
            raise ValueError(f"{kind}: incomplete sources or empty source; expected {sorted(expected)}, got {counts}")


def load_headqa_archive(archive_path):
    """逐字读取官方HEAD_EN三个JSON；图片路径仅作为无图过滤标记，不解压PDF。"""
    split_files = {"train": "train_HEAD_EN.json", "validation": "dev_HEAD_EN.json", "test": "test_HEAD_EN.json"}
    output = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split, filename in split_files.items():
            suffix = f"HEAD_EN/{filename}"
            matches = [name for name in archive.namelist() if name == suffix or name.endswith("/"+suffix)]
            if len(matches) != 1:
                raise ValueError(f"HeadQA archive: required member {suffix} has {len(matches)} matches")
            with archive.open(matches[0]) as stream:
                payload = json.load(stream)
            if not isinstance(payload.get("exams"), dict):
                raise ValueError(f"HeadQA {split}: exams must be an object")
            rows = []
            for exam in payload["exams"].values():
                for question in exam["data"]:
                    image = require_image_marker(question["image"])
                    rows.append({
                        "name": require_text(exam["name"], "exam name").strip(),
                        "year": require_text(exam["year"], "year").strip(),
                        "category": require_text(exam["category"], "category").strip(),
                        "qid": int(question["qid"]), "qtext": require_text(question["qtext"], "qtext").strip(),
                        "ra": int(question["ra"]), "image": image,
                        "answers": [{"aid": int(answer["aid"]),
                                     "atext": require_text(answer["atext"], "atext").strip()}
                                    for answer in question["answers"]],
                    })
            if not rows:
                raise ValueError(f"HeadQA {split}: empty official split")
            output[split] = datasets.Dataset.from_list(rows)
    return datasets.DatasetDict(output)


def require_image_marker(value):
    if not isinstance(value, str):
        raise ValueError("HeadQA image marker must be a string")
    return value.strip() or None


def load_source(spec, cache_dir=None):
    if spec["name"] == "dvilares/head_qa":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec["name"], repo_type="dataset", revision=spec["revision"],
                               filename="data/head-qa-es-en-pdfs.zip")
        return load_headqa_archive(path)
    return datasets.load_dataset(spec["name"], name=spec["config"], revision=spec["revision"], cache_dir=cache_dir)


def add_env_kwargs(row):
    """仅补齐MedRM真实reset接口；保留所有原列、原问题uid及数据协议。"""
    extra = row["extra_info"]
    if not isinstance(extra, dict):
        raise ValueError("extra_info must be an object")
    environment = {
        "question": require_text(extra.get("question"), "extra_info.question"),
        "question_id": require_text(extra.get("uid"), "extra_info.uid"),
        "ground_truth": require_text(row["reward_model"].get("ground_truth"), "ground_truth"),
        "data_source": require_text(row.get("data_source"), "data_source"),
        "extra_info": dict(extra),
    }
    if "env_kwargs" in row and row["env_kwargs"] != environment:
        raise ValueError("existing env_kwargs contradicts original metadata")
    return {**row, "env_kwargs": environment}


def map_validated_row(spec, row, index, source_split, target_split):
    name = spec["name"]
    try:
        if name == "openlifescienceai/medqa":
            raw = row["data"]
            require_text(raw["Question"], "Question")
            validate_choice(raw["Correct Option"], raw["Options"])
        elif name == "dvilares/head_qa":
            require_text(row["qtext"], "qtext")
            answers = row["answers"]
            aids = [str(a["aid"]) for a in answers]
            if len(aids) != len(set(aids)) or str(row["ra"]) not in aids:
                raise ValueError("invalid HeadQA label or duplicate aid")
            ordered = sorted(answers, key=lambda a: int(a["aid"]))
            validate_options({chr(65+i): a["atext"] for i, a in enumerate(ordered)})
        elif name == "openlifescienceai/medmcqa":
            require_text(row["question"], "question")
            options = {chr(65+i): row["op"+key] for i, key in enumerate("abcd")}
            cop = row["cop"]
            if type(cop) is not int or not 0 <= cop < 4:
                raise ValueError(f"invalid medmcqa label {cop!r}")
            validate_choice(chr(65+cop), options)
        elif name == MEDCALC_SOURCE:
            require_text(row["Question"], "Question")
            require_text(row["Patient Note"], "Patient Note")
            require_text(row["Ground Truth Answer"], "label")
        elif name == "JesseLiu/medbulltes5op":
            require_text(row["question"], "question")
            validate_choice(row["answer_idx"], {key: row["choices"+key] for key in "ABCDE"})
        elif name == "TIGER-Lab/MMLU-Pro":
            require_text(row["question"], "question")
            options = {chr(65+i): value for i, value in enumerate(row["options"])}
            validate_choice(row["answer"], options)
            answer_index = row["answer_index"]
            if type(answer_index) is not int or answer_index != ord(row["answer"])-65:
                raise ValueError("MMLU-Pro answer_index contradicts label")
        elif name == "TsinghuaC3I/MedXpertQA":
            require_text(row["question"], "question")
            validate_choice(row["label"], row["options"])
        else:
            raise ValueError(f"unknown source {name}")
        result = spec["map_fn"](name, target_split)(row, index)
        require_text(result["extra_info"]["qid"], "qid")
        if result["extra_info"]["qid"] == "None":
            raise ValueError("qid is missing")
        result["extra_info"]["source_split"] = source_split
        result["extra_info"]["uid"] = f"{name}:{source_split}:{index}"
        return add_env_kwargs(result)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name}/{source_split} row {index}: {error}") from error


def source_counts(dataset):
    return dict(Counter(dataset["data_source"]))


def overlap_audit(train, test):
    """只统计，不扩大作者去重范围。比较字段为作者渲染后的question（含选项）。"""
    train_keys = Counter(normalize_question_text(r["extra_info"]["question"]) for r in train)
    test_keys = Counter(normalize_question_text(r["extra_info"]["question"]) for r in test)
    shared = set(train_keys) & set(test_keys)
    return {
        "train_duplicate_rows_beyond_first": sum(n-1 for n in train_keys.values()),
        "test_duplicate_rows_beyond_first": sum(n-1 for n in test_keys.values()),
        "exact_shared_questions": len(shared),
        "exact_overlap_train_rows": sum(train_keys[k] for k in shared),
        "exact_overlap_test_rows": sum(test_keys[k] for k in shared),
        "short_train_rows_under_16_words": sum(n for k, n in train_keys.items() if len(k.split()) < 16),
        "short_test_rows_under_16_words": sum(n for k, n in test_keys.items() if len(k.split()) < 16),
        "medcalc_train_rows_exempt": source_counts(train).get(MEDCALC_SOURCE, 0),
        "medcalc_test_rows_exempt": source_counts(test).get(MEDCALC_SOURCE, 0),
    }


def prepare_data(output_dir, cache_dir=None, loader=None):
    """完整七源、作者16-gram规则与10k训练cap；整目录原子发布，不覆盖旧结果。"""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"output directory is not empty: {output_dir}")
    train_parts, test_parts, loading = [], [], {}
    for spec in DATASET_SPECS:
        print(f"processing dataset: {spec['name']} @ {spec['revision']}", flush=True)
        dataset = (loader or load_source)(spec, cache_dir=cache_dir)
        loading[spec["name"]] = {}
        for target, requested, parts in [("train", spec["train_splits"], train_parts),
                                          ("test", spec["test_splits"], test_parts)]:
            for split_name, split in select_splits(dataset, requested, spec["name"]):
                raw_count = len(split)
                if spec.get("filter") == "no_image":
                    split = filter_headqa_no_image(split)
                elif spec.get("filter") == "health":
                    split = split.filter(lambda row: (row.get("category") or "").lower() == "health")
                if not len(split):
                    raise ValueError(f"{spec['name']}/{split_name}: source empty after author filtering")
                loading[spec["name"]][split_name] = {"target": target, "raw_rows": raw_count,
                                                     "retained_rows": len(split)}
                mapped = split.map(
                    lambda row, idx: map_validated_row(spec, row, idx, split_name,
                                                       "train" if target == "train" else split_name),
                    with_indices=True, remove_columns=split.column_names, load_from_cache_file=False,
                )
                parts.append(mapped)
    train = datasets.concatenate_datasets(train_parts).shuffle(seed=42)
    test = datasets.concatenate_datasets(test_parts).shuffle(seed=42)
    validate_source_counts(source_counts(train), source_counts(test))
    before_counts = source_counts(train)
    before_audit = overlap_audit(train, test)
    non_medcalc_test = test.filter(lambda row: row["data_source"] != MEDCALC_SOURCE)
    test_ngrams = collect_ngrams_from_questions(non_medcalc_test, n=16)
    train = train.filter(lambda row: row["data_source"] == MEDCALC_SOURCE or not has_ngram_overlap(
        row["extra_info"]["question"], test_ngrams, n=16))
    after_ngram_counts = source_counts(train)
    train, _ = cap_dataset_per_source(train, 10000)
    train_counts, test_counts = source_counts(train), source_counts(test)
    validate_source_counts(train_counts, test_counts)
    manifest = {
        "status": "complete", "protocol": "FaithMed main-table no-SFT", "seed": 42,
        "train_cap_per_source": 10000, "test_limit": None, "sft_exclusion": "not applicable: no SFT",
        "sources": [{k: v for k, v in spec.items() if k != "map_fn"} for spec in DATASET_SPECS],
        "loading_counts": loading, "train_counts_before_ngram": before_counts,
        "train_counts_after_ngram": after_ngram_counts, "train_counts": train_counts,
        "test_counts": test_counts, "train_rows": len(train), "test_rows": len(test),
        "deduplication": {"n": 16, "normalized_field": "extra_info.question (includes rendered options)",
                          "direction": "non-MedCalc test -> non-MedCalc train",
                          "unique_test_ngrams": len(test_ngrams),
                          "removed_train_rows": sum(before_counts.values())-sum(after_ngram_counts.values()),
                          "limitations": ["MedCalc train and test are excluded", "questions shorter than 16 words are uncovered",
                                          "no within-split deduplication", "no semantic or paraphrase matching"],
                          "audit_before": before_audit, "audit_final": overlap_audit(train, test)},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        train.to_parquet(str(staging/"train.parquet"))
        test.to_parquet(str(staging/"test.parquet"))
        from pyarrow.parquet import ParquetFile
        for split, expected in [("train", len(train)), ("test", len(test))]:
            actual = ParquetFile(staging/f"{split}.parquet").metadata.num_rows
            if actual != expected:
                raise ValueError(f"{split}: parquet row count {actual} != {expected}")
        (staging/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (staging/"preprocess.log").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # 仅允许已核验为空的目录；并发写入会显式失败。
        staging.replace(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "train_counts": train_counts, "test_counts": test_counts}, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="FaithMed no-SFT full evaluation preparation")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--save_dir", help="兼容作者入口：实际输出save_dir/output_name")
    destination.add_argument("--outputdir", help="最终原子发布目录")
    parser.add_argument("--output_name", default="medmix")
    parser.add_argument("--cache_dir", help="HF datasets缓存；Hub下载缓存由HF_HOME管理")
    parser.add_argument("--train_limit", action="store_true", help="兼容旧flag；no-SFT协议始终cap10000/source")
    args = parser.parse_args(argv)
    output = args.outputdir if args.outputdir else os.path.join(args.save_dir, args.output_name)
    return prepare_data(output, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
