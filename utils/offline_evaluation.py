import json
from pathlib import Path
from collections import defaultdict


def calculate_metrics(file_paths, output_file="evaluation_results.json"):
    seen_ids = set()
    duplicates = []
    all_problem_ids = []
    results = []

    # Metrics counters
    correct_count = 0  # CORRECT
    incorrect_count = 0  # INCORRECT
    not_attempted_count = 0  # NOT_ATTEMPTED

    json_files = []

    # Collect all JSON files
    for path_str in file_paths:
        p = Path(path_str)

        if not p.exists():
            print(f"WARNING: Path does not exist: {p}")
            continue

        if p.is_file() and p.suffix == '.json':
            json_files.append(p)
            print(f"Found file: {p}")
        elif p.is_dir():
            found = list(p.glob('**/*.json'))
            print(f"Found {len(found)} JSON files in {p}")
            json_files.extend(found)
        else:
            print(f"Path is neither file nor directory: {p}")

    print(f"\nTotal files to process: {len(json_files)}\n")

    if len(json_files) == 0:
        print("ERROR: No JSON files found!")
        return

    for file_path in json_files:
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            print(f"  Found {len(items)} items in this file")

            for idx, item in enumerate(items):
                problem_id = item.get('input_sample', {}).get('id')

                if not problem_id:
                    print(f"    WARNING: Item {idx} has no problem_id")
                    continue

                all_problem_ids.append(problem_id)

                # Check for duplicates
                if problem_id in seen_ids:
                    duplicates.append({
                        "id": problem_id,
                        "file": str(file_path),
                        "item_index": idx
                    })
                    continue

                seen_ids.add(problem_id)

                # Parse grade from eval_result
                grade = item.get('eval_result', {}).get('metrics', {}).get('grade', 'UNKNOWN')
                score = item.get('eval_result', {}).get('score', 0)

                # Count by grade
                if grade == 'CORRECT':
                    correct_count += 1
                elif grade == 'INCORRECT':
                    incorrect_count += 1
                elif grade == 'NOT_ATTEMPTED':
                    not_attempted_count += 1
                else:
                    print(f"    WARNING: Unknown grade '{grade}' for {problem_id}")

                results.append({
                    "id": problem_id,
                    "grade": grade,
                    "score": score,
                    "file": str(file_path)
                })
        except Exception as e:
            print(f"  ERROR: {e}")

    # Calculate metrics
    total = correct_count + incorrect_count + not_attempted_count
    attempted = correct_count + incorrect_count

    # ACC = overall correct / total
    acc = correct_count / total if total > 0 else 0

    # ACC@Att = correct given attempted / attempted
    acc_at_att = correct_count / attempted if attempted > 0 else 0

    # F1 = 2 * (ACC * ACC@Att) / (ACC + ACC@Att)
    f1_score = 2 * (acc * acc_at_att) / (acc + acc_at_att) if (acc + acc_at_att) > 0 else 0

    # Find missing IDs
    id_numbers = set()
    for problem_id in all_problem_ids:
        try:
            num = int(problem_id.split('_')[-1])
            id_numbers.add(num)
        except:
            pass

    missing_ids = []
    if id_numbers:
        min_id = min(id_numbers)
        max_id = max(id_numbers)
        expected_range = set(range(min_id, max_id + 1))
        missing_ids = sorted(list(expected_range - id_numbers))

    print(f"\n{'=' * 60}")
    print(f"DETAILED METRICS:")
    print(f"  Total questions (N): {total}")
    print(f"  Correct (C): {correct_count}")
    print(f"  Incorrect: {incorrect_count}")
    print(f"  Not Attempted: {not_attempted_count}")
    print(f"  Attempted (A): {attempted}")
    print(f"\nCALCULATED METRICS:")
    print(f"  Overall Correct (ACC) = C/N = {correct_count}/{total} = {acc:.4f} ({acc * 100:.2f}%)")
    print(
        f"  Correct Given Attempted (ACC@Att) = C/A = {correct_count}/{attempted} = {acc_at_att:.4f} ({acc_at_att * 100:.2f}%)")
    print(f"  F1-Score = 2×(ACC×ACC@Att)/(ACC+ACC@Att) = {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print(f"\nDUPLICATES REMOVED: {len(duplicates)}")

    if duplicates:
        print(f"\n排除的重复ID (前10个):")
        for dup in duplicates[:10]:
            print(f"  ID: {dup['id']} (from {dup['file']})")
        if len(duplicates) > 10:
            print(f"  ... 还有 {len(duplicates) - 10} 个重复")

    if missing_ids:
        print(f"\nID范围分析:")
        print(f"  最小ID编号: {min(id_numbers)}")
        print(f"  最大ID编号: {max(id_numbers)}")
        print(f"  预期ID数: {max(id_numbers) - min(id_numbers) + 1}")
        print(f"  实际ID数: {len(id_numbers)}")
        print(f"  缺失ID数: {len(missing_ids)}")
        print(f"\n缺失的ID编号 (前20个):")
        for mid in missing_ids[:20]:
            print(f"    simpleqa_verified_{mid}")
        if len(missing_ids) > 20:
            print(f"    ... 还有 {len(missing_ids) - 20} 个缺失ID")

    # Prepare output summary
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
        },
        # "duplicates_removed": len(duplicates),
        # "duplicate_details": duplicates,
        # "missing_ids": missing_ids,
        # "detailed_results": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    file_paths = [
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251017_113155_240_questions/results.json",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251016_142727_500_questions",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251016_234248_250_questions",
        "/mnt/d/src/eigent_plus/librarian/results/simpleqa_verified_eval_agent=eigent_search_model=gpt-4.1-mini_20251017_134814/results.json"
    ]
    calculate_metrics(file_paths)