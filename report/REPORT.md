# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Minh Khải
**Nhóm:** Nhóm 03
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là 2 câu tương đồng về ngữ nghĩa

**Ví dụ HIGH similarity:**
- Sentence A: "Con mèo nằm trên ban công"
- Sentence B: "Trên ban công, con mèo đang nằm"
- Tại sao tương đồng: Cả 2 đều mô tả cùng 1 ý, chỉ khác cách diễn đạt. Vector embedding của chúng sẽ trỏ gần cùng 1 hướng.

**Ví dụ LOW similarity:**
- Sentence A: "Con mèo nằm trên ban công"
- Sentence B: "Cái ô tô màu đen này rất đẹp"
- Tại sao khác: Hai câu này khác hoàn toàn chủ đề, không có overlap về ngữ nghĩa. Vector embedding của chúng trỏ ra 2 hướng rất khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Euclidean distance đo khoảng cách thẳng giữa 2 điểm trong không gian, và nó bị ảnh hưởng bởi độ lớn của vector.
Cosine similarity thì quan tâm đến hướng, không quan tâm đến độ lớn.
Trong NLP, hướng là thứ mang ý nghĩa. Hai câu cùng nghĩa thì trong không gian embedding, 2 vector tương ứng trỏ về cùng 1 hướng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> overlap=50 là chunk này chứa 50 kí tự chunk ngay trước đó.
=> Chunk 1: 500 kí tự. Từ chunk 2, mỗi chunk đóng góp thêm 450 kí tự mới (50 ki tự trùng chunk trước)
=> Có 23 chunk: 10000 = 500 + 450 * 21 + 50
> *Đáp án: 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Chunk 1: 500 kí tự. Từ chunk 2, mỗi chunk đóng góp thêm 400 kí tự mới (100 ki tự trùng chunk trước)
=> Số lượng chunk sẽ tăng lên, do từ chunk thứ 2, chỉ lấy thêm 400 kí tự mới so với 450 kí tự mới với overlap=50
Overlap nhiều hơn giúp giảm khả năng 1 câu quan trọng bị cắt đứt giữa 2 chunk. Overlap lớp, càng nhiều context được giữ lại ở mỗi chunk và AI càng ít bị miss thông tin quan trọng.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Domain: Vietnamese Fairy Tales - Truyện cổ tích Việt Nam

**Tại sao nhóm chọn domain này?**
> Truyện cổ tích có context dài, nội dung phong phú và cấu trúc rõ ràng — phù hợp để kiểm tra khả năng retrieve chính xác của RAG 

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Sọ Dừa | https://loigiaihay.com/so-dua-truyen-co-tich-viet-nam-a185655.html | 5634 | story_title: ["Sọ Dừa"], story_type: "cổ tích", origin: "Việt Nam", themes: ["phép thuật", "tình yêu", "lòng tốt"], main_characters: ["Sọ Dừa", "Phú Ông", "cô út"]|
| 2 | Thạch Sanh | https://loigiaihay.com/thach-sanh-truyen-co-tich-viet-nam-a187017.html | 9207 | story_title: ["Thạch Sanh"], story_type: "cổ tích", origin: "Việt Nam", themes: ["anh hùng", "phép thuật", "thiện ác"], main_characters: ["Thạch Sanh", "Lý Thông", "công chúa"]|
| 3 | Hồ Gươm | https://loigiaihay.com/su-tich-ho-guom-truyen-co-tich-the-gioi-a182987.html | 7017 | story_title: ["Sự tích Hồ Gươm", "Hồ Hoàn Kiếm"], story_type: "truyền thuyết", origin: "Việt Nam", themes: ["lịch sử", "yêu nước", "thần linh"], main_characters: ["Lê Lợi", "Rùa Vàng", "Lê Thận"] |
| 4 | Ngưu Lang Chức Nữ | https://loigiaihay.com/su-tich-nguu-lang-chuc-nu-truyen-co-tich-the-gioi-a183361.html | 5284 | story_title: ["Ngưu Lang Chức Nữ"], story_type: "truyền thuyết", origin: "Trung Quốc", themes: ["tình yêu", "chia ly", "thiên đình"], main_characters: ["Ngưu Lang", "Chức Nữ", "Ngọc Hoàng"] |
| 5 | Cây Khế | https://loigiaihay.com/su-tich-cay-khe-truyen-co-tich-viet-nam-a183775.html | 10086 | story_title: ["Cây Khế"], story_type: "cổ tích", origin: "Việt Nam", themes: ["tham lam", "thiện ác", "lòng tốt"], main_characters: ["người anh", "người em", "chim phượng hoàng"] |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| story_title | list | ["Sọ Dừa"] | Lọc đúng tài liệu khi user hỏi theo tên truyện |
| story_type | string | "cổ tích" / "truyền thuyết" | Phân loại thể loại, filter theo nhóm |
| origin | string | "Việt Nam" / "Trung Quốc" | Phân biệt nguồn gốc truyện |
| main_characters | list | ["Sọ Dừa", "Phú Ông"] | Retrieve tài liệu khi user hỏi về nhân vật cụ thể |
| themes | list | ["phép thuật", "tình yêu"] | Gợi ý truyện cùng chủ đề |

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2 tài liệu "Thạch Sanh" và "Sọ Dừa":

| Tài liệu           | Strategy                         | Chunk Count | Avg Length | Preserves Context?                                                                                                                           |
| ------------------ | -------------------------------- | ----------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| cayke.txt          | FixedSizeChunker (`fixed_size`)  | 47          | 198.2      | Cắt theo số ký tự cứng nhắc. Rất dễ cắt đôi từ hoặc cắt giữa câu, làm mất ý nghĩa.                                                           |
|                    | SentenceChunker (`by_sentences`) | 23          | 303.0      | Giữ được trọn vẹn ý nghĩa của từng câu. Tuy nhiên, mối liên hệ giữa các câu trong cùng một đoạn có thể bị mất.                               |
|                    | RecursiveChunker (`recursive`)   | 54          | 128.0      | Ưu tiên cắt theo đoạn văn (\n\n), sau đó mới đến dòng (\n) và câu. Nó giữ các thông tin có liên quan về mặt cấu trúc ở gần nhau nhất có thể. |
| nguulangchucnu.txt | FixedSizeChunker (`fixed_size`)  | 35          | 199.5      | Xuyên tạc ý nghĩa do cắt vụn giữa các đoạn hội thoại hoặc diễn biến tình cảm quan trọng của Ngưu Lang và Chúc Nữ. |
|                    | SentenceChunker (`by_sentences`) | 13          | 404.4      | Khá tốt, bảo toàn được nội dung trọn vẹn của câu nói mong nhớ, nhưng làm đứt gãy luồng cảm xúc liền mạch giữa 2 câu. |
|                    | RecursiveChunker (`recursive`)   | 41          | 127.0      | Giữ được toàn bộ diễn biến của từng phân cảnh (như cảnh chia ly ở sông Ngân) trong một khối duy nhất. |
| sodua.txt          | FixedSizeChunker (`fixed_size`)  | 38          | 196.9      | Mất ngữ cảnh về sự biến hóa về diện mạo và hành động của Sọ Dừa do chunk bị cắt cụt ở giữa dòng miêu tả. |
|                    | SentenceChunker (`by_sentences`) | 24          | 232.8      | Ổn định, nhưng làm đứt liên kết nguyên nhân - kết quả của câu chuyện. |
|                    | RecursiveChunker (`recursive`)   | 39          | 142.5      | Bao bọc toàn bộ các tình huống phép thuật kì ảo của Sọ Dừa nguyên vẹn trong một chunk. |

### Strategy Của Tôi

**Loại:** [RecursiveChunker]

**Mô tả cách hoạt động:**
> RecursiveChunker chia văn bản theo danh sách separator theo thứ tự ưu tiên: `"\n\n"` → `"\n"` → `". "` → `" "` → `""`. Với mỗi đoạn văn bản, nó thử tách tại separator ưu tiên cao nhất trước — nếu chunk kết quả vẫn vượt quá kích thước cho phép, nó tiếp tục đệ quy xuống separator cấp thấp hơn. 


**Tại sao tôi chọn strategy này cho domain nhóm?**
> Truyện cổ tích được viết theo đoạn văn tự nhiên, mỗi đoạn thường ứng với một tình tiết hoàn chỉnh. RecursiveChunker khai thác điều này bằng cách ưu tiên tách tại `"\n\n"` — giữ nguyên ranh giới đoạn văn gốc, tránh cắt đứt giữa chừng một sự kiện. 

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy         | Retrieval Score (/10) | Điểm mạnh                                                                                               | Điểm yếu                                                                             |
| ---------- | ---------------- | --------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Tôi       | RecursiveChunker | 7.0/10                | Giữ vẹn nguyên độ liền mạch trong cốt truyện của nhân vật, bảo toàn Chunk Coherence tuyệt đối. | Kịch bản khởi tạo phức tạp và tốn CPU xử lý phép toán đệ quy.                        |
| Trí        | FixedSizeChunker | 4.0/10                | Tốc độ chuẩn bị dữ liệu (Index) nhanh nhất, kích cỡ chunk đều đặn dễ cấp phát bộ nhớ.                   | Đánh mất hoàn toàn bối cảnh truyện do ranh giới cắt chữ rơi ngẫu nhiên vào giữa câu. |
| Tuấn       | SentenceChunker  | 6.0/10                | Bắt khá chuẩn các câu "Rút Ra Bài Học" ở cuối truyện do chỉ chứa 1 dấu chấm chấm dứt.                  | Dễ gián đoạn các sự kiện có ngữ cảnh kéo dài (buộc LLM phải đọc nhiều chunk ngắt quãng).     |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker tốt nhất, vì truyện cổ tích được viết chia đoạn tự nhiên, mỗi đoạn thường ứng với một tình tiết hoàn chỉnh.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*

### Test Results

```

(vin) K:\2A202600343-VuMinhKhai-Day07>pytest tests/ -v
================================================= test session starts =================================================
platform win32 -- Python 3.12.13, pytest-9.0.2, pluggy-1.6.0 -- C:\Users\Khai\.conda\envs\vin\python.exe
cachedir: .pytest_cache
rootdir: K:\2A202600343-VuMinhKhai-Day07
plugins: anyio-4.13.0, langsmith-0.7.26
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                            [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                     [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                              [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                               [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                    [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                    [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                          [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                           [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                         [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                           [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                           [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                      [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                  [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                            [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                   [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                       [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                 [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                       [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                           [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                             [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                               [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                     [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                          [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                            [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                       [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                           [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                 [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                           [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED        [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                      [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                     [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED         [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                    [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED             [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED   [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED       [100%]

================================================= 42 passed in 0.69s ==================================================

(vin) K:\2A202600343-VuMinhKhai-Day07>
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | Là một khối thịt đỏ hỏn, không tay không chân, tròn lăn lóc giống như một quả dừa. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | Lý Thông vốn là kẻ tiểu nhân tráo trở, thấy Thạch Sanh thật thà khoẻ mạnh nên lợi dụng để cướp công giết chằn tinh nhằm tiến thân. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | Gắn liền trực tiếp với cuộc chiến của vua Lê Lợi (mệnh danh Bình Định Vương) mượn Gươm của Thần Kim Quy đánh tan quân Minh. |
| 4 | Bi kịch của Ngưu Lang và Chúc Nữ bắt nguồn từ đâu? | Bắt nguồn từ sự cấm cản của Ngọc Hoàng vì ranh giới Tiên - Phàm và trách nhiệm chốn tiên giới bị bỏ bê. |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | Lòng tham vô đáy (như người anh) sẽ chuốc lấy sự hủy diệt, còn sự chia sẻ yêu thương sẽ đơm bông kết trái bền vững. |

### Kết Quả Của Tôi

--- Benchmark Results ---
| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|---|---|---|---|---|
| 1 | Sọ Dừa có ngoại hình như thế nào từ khi sinh ra? | [book/caykhe.txt] Chim mới ăn vài quả, hai vợ chồng đã tru tréo lên: “Cả nhà c... | 0.320 | | Trong phần ngữ cảnh bạn cung cấp, không có thông tin nào về ngoại hình của Sọ Dừa từ khi sinh ra. Tôi không biết. |
| 2 | Vì sao Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại? | [book/thachsanh.txt] Một hôm, có anh hàng rượu tên là Lý Thông đi ngang qua đó gh... | 0.285 | | Thạch Sanh tốt bụng nhưng vẫn bị Lý Thông hãm hại vì Lý Thông thấy Thạch Sanh khỏe mạnh, lại mồ côi, có thể lợi dụng được. Do đó, Lý Thông đã kết nghĩa làm anh em với Thạch Sanh để dễ dàng thao túng và hãm hại anh. |
| 3 | Sự tích Hồ Gươm có liên quan đến vị anh hùng lịch sử nào? | [book/hoguom.txt] Đến lần thứ ba kéo lưới vẫn là lưỡi gươm đó mắc vào. Thấy lạ... | 0.346 | | Sự tích hồ gươm liên quan đến anh hùng lịch sử Lê Lợi |
| 4 | Bi kịch của Ngưu Lang và Chức Nữ bắt nguồn từ đâu? | [book/nguulangchucnu.txt]  Một năm vào mùa thu, chị dâu bắt Ngưu Lang phải đi chăn bò,... | 0.319 | | Do Ngưu Lang và Chức Nữ vi phạm quy tắc Thiên đình dẫn đến việc Ngọc Hoàng nổi giận và chia cắt họ |
| 5 | Bài học rõ nét nhất từ câu chuyện Cây Khế? | [book/caykhe.txt] Những ngày sung sướng nhất của hai vợ chồng người em là nhữn... | 0.247 | | Bài học rõ nét nhất từ câu chuyện Cây Khế là giá trị của lòng yêu thương, sự chăm sóc và sự cần mẫn trong cuộc sống. Hai vợ chồng chăm sóc cây khế tận tình đã gặt hái được những quả ngọt, tượng trưng cho những thành quả đạt được nhờ sự cố gắng và tình yêu đối với công việc và cuộc sống. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 15/ 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 10/ 10 |
| Core implementation (tests) | Cá nhân | 30/ 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
