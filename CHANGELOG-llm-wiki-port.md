# LLM Wiki Feature Port — Changelog

OpenKB에 LLM Wiki의 핵심 기능을 포팅한 Phase 0-3 변경 내역.

## Phase 0: Korean Setup + Entity Typing

**목표**: 한국어 출력 지원 + 개념 분류 체계 도입

### 기능 강화

- **한국어 섹션 헤더**: `openkb init` 시 언어 설정에 따라 index.md 섹션 헤더가 한국어로 생성됨 (`문서`, `개념`, `탐구`)
- **개체 유형 분류**: 개념 생성 시 7가지 entity_type 지정 — `person`, `organization`, `technology`, `concept`, `event`, `location`, `work`
- **프론트매터 확장**: 개념 페이지 frontmatter에 `entity_type` 필드 저장
- **컨셉 플랜 프롬프트 개선**: `_CONCEPTS_PLAN_USER` 프롬프트가 entity_type을 명시적으로 요청하도록 변경

### CLI 변경

```
openkb init          → language: ko 시 한국어 섹션 헤더 생성
openkb add            → 개념 페이지에 entity_type frontmatter 포함
```

### 변경 파일

- `openkb/agent/compiler.py` — 프롬프트, `_write_concept`, `_read_concept_briefs`, `_update_index` 수정
- `openkb/cli.py` — init 명령에 `_make_index_template(language)` 적용

---

## Phase 1: 2-Step Ingest + Review System

**목표**: 분석 단계 추가 + LLM 오류 인간 검토 시스템

### 기능 강화

- **2단계 수집 파이프라인**: 기존 `doc → summary → concepts` 흐름을 `doc → analysis → summary → concepts`로 변경
  - 분석 단계에서 entities, concept_actions, review_items 생성
  - concept_actions가 컨셉 플랜 프롬프트에 "Prior analysis"로 주입되어 중복 개념 생성 방지
- **리뷰 시스템**: LLM 출력에서 `---REVIEW---` 구분자로 리뷰 항목 추출
  - `ReviewItem` 데이터클래스: type, title, description, source_path, affected_pages, search_queries, options
  - 타입 검증: contradiction, duplicate, missing_page, confirm, suggestion
  - `.openkb/review_queue.json` 영속 저장
- **CLI 리뷰 명령**: 대기 중인 리뷰 항목 조회, 수락, 건너뛰기

### CLI 변경

```
openkb add            → 분석 단계 선행 실행, review_queue.json 자동 생성
openkb review         → 대기 중인 리뷰 항목 목록
openkb review --accept 1  → 항목 1 수락
openkb review --skip 2   → 항목 2 건너뛰기
```

### 변경 파일

- `openkb/agent/compiler.py` — `_analyze_document`, `_ANALYSIS_USER` 프롬프트, 파이프라인 수정
- `openkb/review/__init__.py` (신규)
- `openkb/review/models.py` (신규) — ReviewItem 데이터클래스
- `openkb/review/parser.py` (신규) — ---REVIEW--- 파서
- `openkb/review/queue.py` (신규) — ReviewQueue 영속 관리
- `openkb/cli.py` — review 서브커맨드 추가

---

## Phase 2: Graph Engine + Insights

**목표**: 위키링크 그래프 구축 + 커뮤니티 탐지 + 지식 갭 발굴

### 기능 강화

- **그래프 구축** (`openkb/graph/build.py`): wiki/ 디렉토리의 모든 .md 파일에서 `[[wikilink]]`와 frontmatter(sources, entity_type)를 추출하여 networkx Graph 생성
  - 위키링크 엣지: edge_type=wikilink, weight=1
  - 소스 중복 엣지: edge_type=source_overlap, weight=공유 소스 수
  - 존재하지 않는 위키링크 대상은 placeholder 노드로 등록(고립 노드로 표시)
  - 직렬화: `.openkb/graph.json` (nodes, edges, metadata)
- **4-시그널 Relevance** (`openkb/graph/relevance.py`): 두 노드 간 관련도 점수 계산
  - direct_link (3.0) + source_overlap (4.0×공유수) + adamic_adar (1.5×Σ1/log(deg)) + type_affinity (1.0)
  - `top_related(graph, seed, k)`: seed 노드의 top-k 관련 노드 반환
- **커뮤니티 탐지** (`openkb/graph/community.py`): Louvain 알고리즘으로 커뮤니티 분할
  - cohesion = 내부 엣지 밀도 / 최대 가능 엣지 수
  - cohesion < 0.15인 커뮤니티를 "sparse"로 플래깅
- **인사이트** (`openkb/graph/insights.py`): 4가지 지식 인사이트 생성
  - 고립 노드: degree ≤ 1
  - 희소 커뮤니티: cohesion < 0.15
  - 브릿지 노드: 3+개 커뮤니티에 연결된 노드
  - 놀라운 연결: cross-community 엣지, type-variant 엣지 (서로 다른 entity_type)
- **자동 그래프 리빌드**: `openkb add` 완료 시 compiler.py에서 `build_and_save_graph` 자동 호출
- **CLI 인사이트 명령**: 그래프 기반 지식 갭 리포트 출력

### CLI 변경

```
openkb add            → graph.json 자동 리빌드
openkb insights       → 그래프 인사이트 리포트 출력
```

insights 출력 예시:
```
Graph: 45 nodes, 78 edges, 5 communities

Surprising Connections:
  "regulation" ↔ "innovation" (cross-community, relevance: 8.20)
  "EU AI Act" ↔ "startup" (type-variant, relevance: 6.10)

Knowledge Gaps:
  Orphan nodes: quantum-computing (1 edge)
  Sparse communities: robotics (cohesion: 0.08)
  Bridge nodes: openai (3 communities)

Communities:
  AI/ML (28 pages, cohesion: 0.420)
  Regulation/Policy (15 pages, cohesion: 0.310)
```

### 변경 파일

- `openkb/graph/__init__.py` (신규)
- `openkb/graph/build.py` (신규) — 그래프 구축 + 직렬화
- `openkb/graph/relevance.py` (신규) — 4-시그널 relevance
- `openkb/graph/community.py` (신규) — Louvain + cohesion
- `openkb/graph/insights.py` (신규) — 인사이트 생성
- `openkb/agent/compiler.py` — `_compile_concepts`에 자동 리빌드 후킹
- `openkb/cli.py` — insights 서브커맨드 추가
- `pyproject.toml` — networkx, python-louvain 의존성 추가

---

## Phase 3: Hybrid Search

**목표**: 그래프 기반 검색 확장으로 간접 관련 페이지 발견

### 기능 강화

- **search_related 도구**: query agent에 새로운 tool 추가
  - `search_related(page_name, top_k)`: graph.json에서 relevance score 기반으로 관련 페이지 반환
  - graph.json 미존재 시 안내 메시지 반환
  - 페이지 미존재 시 not-found 메시지 반환
  - 결과 포맷: `page_slug (relevance: X.XX)`
- **쿼리 에이전트 지침 업데이트**: 키워드 매칭 후 search_related 호출로 간접 관련 페이지 발견 권장
- **LanceDB 불필요**: 그래프 확장만으로 "시맨틱 부스트" 제공 — 벡터 임베딩 없이 relevance score 기반

### CLI 변경

```
openkb query "AI regulation"  → 키워드 매칭 + 그래프 확장 결과 포함
```

쿼리 에이전트 도구 목록 (4개):
1. `read_file` — wiki 파일 읽기
2. `get_page_content` — 긴 문서 특정 페이지 읽기
3. `get_image` — 이미지 보기
4. `search_related` — 그래프 기반 관련 페이지 검색 (신규)

### 변경 파일

- `openkb/agent/tools.py` — `search_related_pages` 평면 함수 추가
- `openkb/agent/query.py` — `search_related` function_tool 추가, 지침 업데이트

---

## 코드 품질 개선 (Convergence + Improve Loop)

### 공유 유틸리티 추출

- `openkb/json_utils.py` (신규): `extract_json()` — 괄호 깊이 추적 기반 JSON 추출 공유 유틸리티
  - `openkb/review/parser.py`의 `_extract_json_array`와 `openkb/agent/compiler.py`의 `_parse_json`에서 중복 로직 제거
  - `_parse_json`은 `json_repair` 폴백 유지

### 컴파일러 프론트매터 헬퍼

- `_split_frontmatter(text) → (frontmatter, body)` — frontmatter 분할
- `_inject_fm_field(frontmatter, field, value)` — YAML 필드 삽입/교체
- `_strip_frontmatter(text)` — frontmatter 블록 제거
- `_write_concept` 73줄 → 47줄로 감소

### CLI 재시도 헬퍼

- `_compile_with_retry(fn, attempts=2, delay=2.0)` — 컴파일 재시도 공통 로직
- `compile_short_doc`/`compile_long_doc` 중복 for-attempt 패턴 제거

### 그래프 모듈 품질

- **build.py**: YAML sources 파싱 버그 수정 (`in_sources_list` 상태 변수 도입)
- **build.py**: `load_graph` 손상 JSON 예외 처리 추가
- **community.py**: `compute_cohesion` O(E) → O(k) 최적화 (subgraph 사용)
- **insights.py**: `_build_node_to_comm` 헬퍼로 node_to_comm 중복 빌드 제거
- **__init__.py**: 공개 API 전체 재export

---

## 모듈 규모 요약

| 모듈 | 줄 수 | 신규/수정 |
|------|-------|-----------|
| openkb/agent/compiler.py | 1,033 | 수정 |
| openkb/agent/query.py | 239 | 수정 |
| openkb/agent/tools.py | 226 | 수정 |
| openkb/graph/build.py | 226 | 신규 |
| openkb/graph/relevance.py | 63 | 신규 |
| openkb/graph/community.py | 50 | 신규 |
| openkb/graph/insights.py | 151 | 신규 |
| openkb/review/models.py | 48 | 신규 |
| openkb/review/parser.py | 46 | 신규 |
| openkb/review/queue.py | 55 | 신규 |
| openkb/json_utils.py | 83 | 신규 |
| openkb/cli.py | 964 | 수정 |
| **합계** | **3,184** | |

## 의존성 추가

```toml
dependencies = [
    # ... 기존 ...
    "networkx>=3.0",          # 그래프 구축 및 알고리즘
    "python-louvain>=0.16",   # Louvain 커뮤니티 탐지
]
```

## Phase 의존성

```
Phase 0 (Korean + entity type)
    ↓
Phase 1 (2-step ingest + review)
    ↓  (entity_type → type_affinity 신호에 필요)
Phase 2 (graph engine + insights)
    ↓  (graph.json → 검색 확장에 필요)
Phase 3 (hybrid search)
    ↓
Phase 4 (pkm-bot integration — 별도 리포)
```