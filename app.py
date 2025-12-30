import streamlit as st
import base64
import os
import requests
import re
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ============================================================
# 페이지 및 기본 설정
# ============================================================
st.set_page_config(
    page_title="새싹 스마트 AI 취업 컨설턴트",
    page_icon="🤖",
    layout="wide",
)

# Document 폴더 자동 생성
if not os.path.exists("Document"):
    os.makedirs("Document")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ============================================================
# 커스텀 CSS (All-White & Clean Blue 테마)
# ============================================================
st.markdown(
    """
<style>
    .stApp { background-color: #ffffff; }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f2f6;
    }

    .user-box {
        background-color: #0066cc; 
        color: white; 
        padding: 15px;
        border-radius: 20px 20px 5px 20px; 
        margin: 10px 0 10px 20%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 15px;
    }
    .ai-box {
        background-color: #f8f9fa; 
        color: #1a1a1a; 
        padding: 15px;
        border-radius: 20px 20px 20px 5px; 
        margin: 10px 20% 10px 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        font-size: 15px;
    }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #0066cc;
        background-color: white;
        color: #0066cc;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        color: white;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-color: #e9ecef !important;
    }
    
    .search-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .source-link {
        color: #0066cc;
        font-size: 0.9em;
    }
    
    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .mode-rag {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .mode-web {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .mode-llm {
        background-color: #fff3e0;
        color: #e65100;
    }
</style>
""",
    unsafe_allow_html=True,
)


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None


# ============================================================
# RAG: 인덱싱 함수
# ============================================================
def perform_indexing():
    with st.spinner("Document 폴더 내 문서를 인덱싱 중입니다..."):
        try:
            loader = PyPDFDirectoryLoader("Document/")
            documents = loader.load()
            if not documents:
                st.warning("Document 폴더에 PDF 파일이 없습니다.")
                return
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vector_store = vectorstore
            st.success(f"인덱싱 완료! 총 {len(splits)}개의 지식 조각을 생성했습니다.")
        except Exception as e:
            st.error(f"인덱싱 중 오류 발생: {e}")


# ============================================================
# 웹 검색 함수
# ============================================================
def search_naver_blog(query: str, num_results: int = 10) -> list:
    """네이버 블로그 검색 API"""
    url = "https://openapi.naver.com/v1/search/blog.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {
        "query": query,
        "display": num_results,
        "sort": "sim",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 블로그",
                    "date": item.get("postdate", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_naver_cafe(query: str, num_results: int = 10) -> list:
    """네이버 카페 검색 API"""
    url = "https://openapi.naver.com/v1/search/cafearticle.json"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
    }
    params = {"query": query, "display": num_results, "sort": "sim"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_results = []
        for item in results.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            description = re.sub(r"<[^>]+>", "", item.get("description", ""))
            search_results.append(
                {
                    "title": title,
                    "link": item.get("link", ""),
                    "snippet": description,
                    "source": "네이버 카페",
                    "cafe_name": item.get("cafename", ""),
                }
            )
        return search_results
    except Exception as e:
        return []


def search_web(query: str, sources: list, num_results: int = 5) -> list:
    """네이버 블로그 + 카페 통합 검색"""
    all_results = []
    if "네이버 블로그" in sources:
        all_results.extend(search_naver_blog(query, num_results))
    if "네이버 카페" in sources:
        all_results.extend(search_naver_cafe(query, num_results))
    return all_results


# ============================================================
# 질문 분류 함수
# ============================================================
def classify_query(query: str, has_vector_store: bool) -> str:
    """
    질문을 분류하여 RAG / LLM / 웹 검색으로 분기
    1. SeSAC, 새싹, 교육 관련 → RAG
    2. 그 외 → LLM이 판단 (AUTO)
    """
    # SeSAC/교육 관련 키워드 (RAG 사용)
    rag_keywords = ["새싹", "SeSAC", "성동", "캠퍼스", "교육과정", "수강후기", "교육성과", "장한평", "답십리"]
    
    query_lower = query.lower()
    
    # RAG 키워드 체크
    for keyword in rag_keywords:
        if keyword in query_lower:
            return "RAG"
    
    # 그 외 질문은 LLM이 자동 판단하도록 AUTO 반환
    return "AUTO"


def determine_search_need(query: str, api_key: str) -> dict:
    """
    LLM을 사용하여 질문이 웹 검색이 필요한지 판단
    Returns: {"need_search": bool, "reason": str, "search_query": str}
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=api_key,
        temperature=1,
    )
    
    classification_prompt = f"""당신은 질문 분류기입니다. 반드시 JSON 형식으로만 응답하세요.

[웹 검색이 필요한 질문 유형]
- 채용 공고, 신입/경력 모집 소식, 채용 사이트(원티드, 사람인 등) 정보
- 특정 기업의 직무별 자격 요건 및 우대 사항
- 면접 후기, 기업 문화, 연봉 정보 등 실시간 리뷰

[웹 검색이 필요 없는 질문 유형]
- 일반 지식, 개념 설명
- 코딩, 프로그래밍 도움
- 수학, 과학 등 보편적 지식
- 번역, 문법 교정
- 창작, 글쓰기
- 일반적인 조언

질문: "{query}"

위 질문을 분석하여 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요:
{{"need_search": true, "reason": "이유", "search_query": "검색어"}}
또는
{{"need_search": false, "reason": "이유", "search_query": ""}}"""
    
    try:
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        result_text = response.content.strip()
        
        # ```json 등의 마크다운 제거
        if "```" in result_text:
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            result_text = result_text.strip()
        
        # JSON 파싱 시도
        result = json.loads(result_text)
        
        # 필수 키 검증
        if "need_search" not in result:
            result["need_search"] = False
        if "reason" not in result:
            result["reason"] = "자동 판단"
        if "search_query" not in result:
            result["search_query"] = ""
            
        return result
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트에서 판단 시도
        result_lower = response.content.lower() if response else ""
        if "true" in result_lower or "필요" in result_lower:
            return {"need_search": True, "reason": "웹 검색 필요로 판단", "search_query": query}
        return {"need_search": False, "reason": "AI 직접 답변 가능", "search_query": ""}
    except Exception as e:
        # 기타 오류 시 기본값 반환
        return {"need_search": False, "reason": f"판단 중 오류: {str(e)}", "search_query": ""}


# ============================================================
# 대표 질문용 미리 정의된 답변
# ============================================================
PREDEFINED_ANSWERS = {
    "🎯 직무 역량 분석법": """
이 공고의 핵심 역량은?

**💡 이렇게 질문해 보세요!**
> 💬 "이 공고의 핵심 기술 스택 3가지와 그 이유를 알려줘."

* **직무 스캔**: 공고 내 핵심 역량 정밀 추출
* **우선순위**: 필수 요건과 우대 사항 완벽 구분
* **지원 전략**: 본인의 경험 중 강조할 포인트 제안

**➡️ PDF 업로드(인덱싱) 후 질문하면 더 정확합니다.**
    """,
    "💡 면접 대비 방법": """
실전 면접 준비가 필요하신가요?

**💡 이렇게 질문해 보세요!**
> 💬 "이 공고 기반 예상 질문 5개와 합격 답변 키워드 알려줘."

* **맞춤 질문**: JD 기반 실무/인성 면접 문항 생성
* **답변 가이드**: 논리적인 답변을 위한 핵심 가이드 제공
* **실전 연습**: "나랑 면접 연습하자"라고 말해보세요!

**➡️ 공고 분석 후 요청하시면 가장 날카로운 질문이 나옵니다.**
    """,
    "📊 연봉/트렌드 확인법": """
업계 트렌드와 연봉이 궁금하다면?

**💡 이렇게 질문해 보세요!**
> 💬 "이 업계 신입 연봉 수준과 최신 채용 트렌드 알려줘."

* **연봉 데이터**: 실시간 웹 검색 기반 처우 파악
* **기술 동향**: 현재 업계에서 핫한 기술과 자격증 분석
* **기업 분석**: 면접 후기와 기업 분위기 종합 요약

**➡️ 실시간 검색을 통해 가장 최신 정보를 가져옵니다.**
    """,
}
   

# ============================================================
# 사이드바
# ============================================================
with st.sidebar:
    logo_b64 = get_base64_image("kirby-puffy.png")
    if logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{logo_b64}" width="100%">',
            unsafe_allow_html=True,
        )
    else:
        st.title("🤖 새싹 스마트 AI 취업 컨설턴트")

    st.divider()
    
    # 지식 데이터베이스 섹션
    st.subheader("📚 지식 데이터베이스")
    if st.button("문서 인덱싱 시작"):
        perform_indexing()
    if st.session_state.vector_store:
        st.caption("✅ 문서 학습 완료")

    st.divider()
    
    # 웹 검색 설정 섹션
    st.subheader("🔍 웹 검색 설정")
    search_sources = st.multiselect(
        "검색 소스",
        ["네이버 블로그", "네이버 카페"],
        default=["네이버 블로그", "네이버 카페"],
    )
    num_results = st.slider("소스별 검색 결과 수", 3, 15, 5)
    
    st.divider()
    
# AI 페르소나 설정
    st.subheader("AI 페르소나 설정")
    system_instruction = st.text_area(
        "AI 역할 정의:",
        value="""너는 IT 채용 전문 헤드헌터이자 커리어 컨설턴트야. 
사용자가 채용 정보를 물어보면 [Context]나 웹 검색 결과를 바탕으로 
[직무 개요], [자격 요건], [우대 사항]으로 깔끔하게 정리해서 알려주고, 
해당 직무에 합격하기 위한 커리어 조언도 한 줄 덧붙여줘.""",
        height=150,
    )

    
    st.divider()
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()
    
    # 통계 표시
    st.divider()
    st.subheader("📊 사용 통계")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("대화 수", len(st.session_state.messages) // 2)
    with col2:
        st.metric("웹 검색", len(st.session_state.search_history))

# ============================================================
# 메인 화면
# ============================================================
st.markdown(
    "<h2 style='color: #0066cc;'>새싹 스마트 AI 취업 컨설턴트</h2>", unsafe_allow_html=True
)
st.caption("🚀 AI 취업 컨설턴트 | PDF 공고 분석부터 최신 채용 트렌드 검색까지, 당신만의 합격 전략을 설계합니다.")

st.markdown("### 💡 무엇을 물어봐야 할지 모르겠다면? 클릭해서 가이드를 확인하세요!")
col1, col2, col3 = st.columns(3)
q1 = "🎯 직무 역량 분석법"
q2 = "💡 면접 대비 방법"
q3 = "📊 연봉/트렌드 확인법"

clicked_q = None
if col1.button("🎯 직무 역량 분석법"):
    clicked_q = q1
if col2.button("💡 면접 대비 방법"):
    clicked_q = q2
if col3.button("📊 연봉/트렌드 확인법"):
    clicked_q = q3

st.divider()

# 대화 기록 표시
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(
            f'<div class="user-box">{msg.content}</div>', unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        st.markdown(f'<div class="ai-box">{msg.content}</div>', unsafe_allow_html=True)

user_input = st.chat_input("질문을 입력해주세요. (예: 채용공고 분석, 면접 대비법 등)")
final_query = clicked_q if clicked_q else user_input

if final_query:
    st.markdown(f'<div class="user-box">{final_query}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(HumanMessage(content=final_query))

    # 답변 생성 로직
    if final_query in PREDEFINED_ANSWERS:
        # 미리 정의된 답변
        ai_content = PREDEFINED_ANSWERS[final_query]
        mode_badge = '<span class="mode-badge mode-rag">📚 사전 정의 답변</span>'
    else:
        # 질문 분류
        query_type = classify_query(final_query, st.session_state.vector_store is not None)
        
        try:
            if query_type == "RAG":
                # RAG 모드 (SeSAC/교육 관련)
                mode_badge = '<span class="mode-badge mode-rag">📚 RAG 모드 (교육 정보)</span>'
                
                context = ""
                if st.session_state.vector_store:
                    docs = st.session_state.vector_store.similarity_search(final_query, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                llm = ChatOpenAI(
                    model="gpt-5-mini",
                    api_key=st.secrets["OPENAI_API_KEY"],
                    streaming=True,
                    temperature=1,
                )

                full_system_prompt = f"{system_instruction}\n\n[Context]\n{context if context else '관련 문서 없음'}"
                prompt = [
                    SystemMessage(content=full_system_prompt)
                ] + st.session_state.messages

                with st.spinner("답변 생성 중..."):
                    response = llm.invoke(prompt)
                    ai_content = response.content
                    
            else:
                # AUTO 모드: LLM이 웹 검색 필요 여부 판단
                with st.spinner("질문 분석 중..."):
                    search_decision = determine_search_need(final_query, st.secrets["OPENAI_API_KEY"])
                
                if search_decision["need_search"]:
                    # 웹 검색 모드
                    mode_badge = '<span class="mode-badge mode-web">🔍 웹 검색 모드</span>'
                    
                    search_query = search_decision["search_query"] if search_decision["search_query"] else final_query
                    
                    with st.status(f"🔍 웹에서 '{search_query}' 검색 중...", expanded=True) as status:
                        all_results = []
                        seen_links = set()
                        
                        # 검색 실행
                        results = search_web(search_query, search_sources, num_results)
                        
                        for result in results:
                            if result["link"] not in seen_links:
                                seen_links.add(result["link"])
                                all_results.append(result)
                        
                        st.write(f"✅ {len(all_results)}개의 결과를 찾았습니다.")
                        st.caption(f"💡 판단 이유: {search_decision['reason']}")
                        status.update(label="검색 완료!", state="complete")
                    
                    # 검색 결과 표시
                    if all_results:
                        with st.expander("📑 검색된 원본 자료 보기", expanded=False):
                            for i, result in enumerate(all_results[:10], 1):
                                st.markdown(
                                    f"""
                                <div class="search-result">
                                    <strong>{i}. {result['title']}</strong><br>
                                    <span class="source-link">🔗 <a href="{result['link']}" target="_blank">{result['source']}</a></span><br>
                                    <small>{result['snippet'][:200]}...</small>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                        
                        # 검색 기록 저장
                        st.session_state.search_history.append({
                            "query": search_query,
                            "results_count": len(all_results),
                        })
                    
                    # 웹 검색 결과를 컨텍스트로 구성
                    web_context = ""
                    for i, result in enumerate(all_results, 1):
                        web_context += f"\n[결과 {i}]\n"
                        web_context += f"제목: {result['title']}\n"
                        web_context += f"출처: {result['source']}\n"
                        web_context += f"링크: {result['link']}\n"
                        web_context += f"내용: {result['snippet']}\n"
                    
                    # LLM으로 웹 검색 결과 분석
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    web_system_prompt = f"""{system_instruction}

아래는 사용자 질문과 관련된 웹 검색 결과입니다. 이 정보를 바탕으로 종합적으로 분석하여 답변해주세요.
답변 시 출처 링크를 함께 표시해주세요.

[웹 검색 결과]
{web_context if web_context else '검색 결과 없음'}"""

                    prompt = [
                        SystemMessage(content=web_system_prompt)
                    ] + st.session_state.messages
                    
                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                else:
                    # 일반 LLM 모드 (웹 검색 불필요)
                    mode_badge = '<span class="mode-badge" style="background-color:#fff3e0;color:#e65100;">🧠 AI 직접 답변</span>'
                    
                    llm = ChatOpenAI(
                        model="gpt-5-mini",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        temperature=1,
                    )
                    
                    # 일반 답변용 시스템 프롬프트 (웹 검색 언급 제거)
                    general_system_prompt = "너는 친절하고 유능한 AI 어시스턴트야. 사용자의 질문에 정확하고 도움이 되는 답변을 제공해줘."

                    prompt = [
                        SystemMessage(content=general_system_prompt)
                    ] + st.session_state.messages

                    with st.spinner("답변 생성 중..."):
                        response = llm.invoke(prompt)
                        ai_content = response.content
                    
        except Exception as e:
            ai_content = f"오류가 발생했습니다: {e}"
            mode_badge = '<span class="mode-badge" style="background-color:#ffebee;color:#c62828;">⚠️ 오류</span>'

    # 답변 표시
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box">{ai_content}</div>', unsafe_allow_html=True)
    st.session_state.messages.append(AIMessage(content=ai_content))

# 하단 안내
st.divider()
st.caption(
    """
💡 **사용 안내**: 

✅ **SeSAC 교육 정보 (RAG 모드)**
- 교육과정 안내, 수강 후기, 성동캠퍼스 이용 가이드 등
- 사이드바에서 **[문서 인덱싱]** 완료 시 첨부된 가이드북 기반으로 정확하게 답변합니다.

✅ **기업 공고 및 취업 정보 (웹 검색 모드)**
- 특정 기업(토스, 현대차 등)의 실시간 채용 공고 및 직무 분석
- 최신 연봉 정보, 면접 후기, 업계 트렌드 뉴스 등
- AI가 질문을 분석하여 **🔍 실시간 웹 검색**을 통해 최신 정보를 가져옵니다.

✅ **일반 지식 및 컨설팅 (AI 직접 답변)**
- 자소서 첨삭 가이드, 면접 답변 구조화(STAR 기법), 일반적인 IT 개념 설명 등
- AI의 학습된 지식을 바탕으로 즉시 최적의 답변을 생성합니다.
"""
)