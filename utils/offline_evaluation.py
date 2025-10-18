import json
from pathlib import Path


def calculate_metrics(file_paths, output_file="evaluation_results.json"):
    seen_ids = set()
    duplicates = []
    all_problem_ids = []

    correct_count = 0
    incorrect_count = 0
    not_attempted_count = 0

    json_files = []

    for path_str in file_paths:
        p = Path(path_str)

        if not p.exists():
            continue

        if p.is_file() and p.suffix == '.json':
            json_files.append(p)
        elif p.is_dir():
            found = list(p.glob('**/*.json'))
            json_files.extend(found)

    if len(json_files) == 0:
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]

            for idx, item in enumerate(items):
                problem_id = item.get('input_sample', {}).get('id')

                if not problem_id:
                    continue

                all_problem_ids.append(problem_id)

                if problem_id in seen_ids:
                    duplicates.append({
                        "id": problem_id,
                        "file": str(file_path),
                        "item_index": idx
                    })
                    continue

                seen_ids.add(problem_id)

                grade = item.get('eval_result', {}).get('metrics', {}).get('grade', 'UNKNOWN')
                score = item.get('eval_result', {}).get('score', 0)

                if grade == 'CORRECT':
                    correct_count += 1
                elif grade == 'INCORRECT':
                    incorrect_count += 1
                elif grade == 'NOT_ATTEMPTED':
                    not_attempted_count += 1

        except Exception as e:
            pass

    total = correct_count + incorrect_count + not_attempted_count
    attempted = correct_count + incorrect_count

    acc = correct_count / total if total > 0 else 0
    acc_at_att = correct_count / attempted if attempted > 0 else 0
    f1_score = 2 * (acc * acc_at_att) / (acc + acc_at_att) if (acc + acc_at_att) > 0 else 0

    summary = {
        "metrics": {
            "total_questions": total,
            "correct": correct_count,
            "incorrect": incorrect_count,
            "not_attempted": not_attempted_count,
            "attempted": attempted,
            "overall_correct_acc": round(acc, 4),
            "correct_given_attempted_acc_at_att": round(acc_at_att, 4),
            "f1_score": round(f1_score, 4),
            "overall_correct_percentage": f"{acc * 100:.2f}%",
            "correct_given_attempted_percentage": f"{acc_at_att * 100:.2f}%",
            "f1_score_percentage": f"{f1_score * 100:.2f}%"
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    file_paths = [
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251017_113155_240_questions/results.json",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251016_142727_500_questions",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251016_234248_250_questions",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251017_134814/results.json"
    ]
    calculate_metrics(file_paths)