"""공개 한국어 100문장 선별 도구의 네트워크 없는 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from evaluation.prepare_open_korean_100 import (  # noqa: E402
    Candidate,
    SOURCES,
    build_candidates,
    classify_and_score,
    extract_audio_url,
    select_candidates,
)


class OpenKoreanDatasetTest(unittest.TestCase):
    """일상 문장 점수와 중복 없는 선별 계약을 검증한다."""

    def test_daily_sentence_scores_above_political_news(self) -> None:
        """일상 어휘 문장은 정치 기사보다 높은 우선순위를 갖는다."""

        daily = classify_and_score(
            "오늘 저녁에는 가족과 함께 식당에서 맛있는 음식을 먹었습니다"
        )
        news = classify_and_score(
            "오늘 국회에서 정부 위원회와 장관이 공식 발표를 진행했습니다"
        )
        self.assertIsNotNone(daily)
        self.assertIsNotNone(news)
        assert daily is not None and news is not None
        self.assertGreater(daily[1], news[1])

    def test_non_daily_or_invalid_length_is_rejected(self) -> None:
        """생활 어휘가 없거나 너무 짧은 문장은 후보에서 제외한다."""

        self.assertIsNone(classify_and_score("은하계의 질량 분포를 분석하였다"))
        self.assertIsNone(classify_and_score("친구"))

    def test_audio_url_and_candidate_fields_are_extracted(self) -> None:
        """Dataset Viewer의 Audio 배열과 정답 필드를 해석한다."""

        source = SOURCES[0]
        entries = [
            {
                "row_idx": 7,
                "row": {
                    "id": "sample-7",
                    "speaker_id": 42,
                    "text": "친구를 만나서 함께 커피를 마시며 이야기를 나누었습니다",
                    "audio": [{"src": "https://example.com/audio.flac"}],
                },
            }
        ]
        candidates = build_candidates(source, entries)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].row_index, 7)
        self.assertEqual(candidates[0].speaker_id, "42")
        self.assertEqual(
            extract_audio_url(entries[0]["row"]["audio"]),
            "https://example.com/audio.flac",
        )

    def test_selection_returns_unique_text_and_spreads_topics(self) -> None:
        """선택 결과는 문장 중복을 제거하고 정확한 개수를 반환한다."""

        source = SOURCES[0]
        topics = ["family_social", "food_shopping", "health", "leisure"]
        candidates = []
        for index in range(12):
            candidates.append(
                Candidate(
                    source=source,
                    row_index=index,
                    source_id=str(index),
                    speaker_id=str(index % 4),
                    gender=None,
                    text=f"오늘 친구와 일상 이야기를 나눈 고유 문장 {index}",
                    audio_url=f"https://example.com/{index}.flac",
                    topic=topics[index % len(topics)],
                    score=float(20 - index),
                )
            )
        # 점수가 더 높아도 같은 문장은 한 번만 선택되어야 한다.
        candidates.append(
            Candidate(
                source=source,
                row_index=99,
                source_id="duplicate",
                speaker_id="9",
                gender=None,
                text=candidates[0].text,
                audio_url="https://example.com/duplicate.flac",
                topic="family_social",
                score=100.0,
            )
        )

        selected = select_candidates(candidates, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({item.text for item in selected}), 8)
        self.assertGreaterEqual(len({item.topic for item in selected}), 3)


if __name__ == "__main__":
    unittest.main()
