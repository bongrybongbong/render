from __future__ import annotations
import numpy as np
import pandas as pd
import datetime
import json
from streamlit_markmap import markmap
from page_utils import styled_hashtag, styled_text, render_markdown
from data.first_data import name, meeting_data, data
from datetime import datetime
from PIL import Image
import re
from typing import Optional, Tuple, List, Union, Literal
import base64
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import os
import openai
import graphviz
from dataclasses import dataclass, asdict
from textwrap import dedent
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv
from streamlit_pills import pills
from streamlit_modal import Modal



CATEGORY_NAMES = {
    "maps": "날짜 픽스",
    "widgets": "비즈니스 모델",  # 35
    "charts": "사용자 경험",  # 16
    "image": "UI/UX디자인 ",  # 10
    "video": "기술개발",  # 6
    "text": "테스트계획",  # 12
    "sc": "더보기...",  # 3 
}
CATEGORY_ICONS = [
    "📆",
    "💵",
    "🧰",
    "🌠",
    "💻",
    "🎯",
    "➕",
]

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

predefined_text = "ThinkWide, 기술개발, 하드웨어 호환, VR 헤드셋 지원, 컨트롤러 옵션, 소프트웨어 인터랙션, 손동작 인식, 음성 명령, 개발도구 통합, Unity, Unreal Engine 플러그인, 네트워킹 및 안정성, 멀티플레이어 지원, 데이터 동기화, 플랫폼 구성, 가상 현실 공간 설계, 공간의 레이아웃, 상호작용 가능한 객체, 3D 모델링 및 아바타 시스템, 사용자 정의 아바타, 표정 및 몸짓 표현, 데이터 관리 및 개인화 설정, 사용자 정보 저장, 마인드맵 설정 기억, UI/UX 디자인, 인터페이스 디자인, 직관적인 메뉴, 가이드와 튜토리얼, 접근성 및 사용성 테스트, 다양한 사용자 테스트, 피드백 반영, 비즈니스 모델, 시장 조사 및 경쟁 분석, 타겟 시장 정의, 경쟁 서비스 분석, 가격 책정 및 수익 모델, 구독 모델, 일회성 구매, 프리미엄 기능 설정, 마케팅 전략 및 브랜드 구축, SNS, 인플루언서 마케팅, 브랜딩 자료 개발"

COLOR = "cyan"
FOCUS_COLOR = "#f96c6c"

load_dotenv()
openai.api_key = os.getenv('openai.api_key')

@dataclass
class Message:
    content: str
    role: Literal["user", "system", "assistant"]

    # is a built-in method for dataclasses
    # called after the __init__ method
    def __post_init__(self):
        self.content = self.content.strip()

START_CONVERSATION = [
    Message("""
        You are a useful mind map/undirected graph-generating AI that can generate mind maps
        based on analyzes the correlation of a provided list of words.
    """, role="system"),
    Message("""
            I'll give you a list of words from brainstorming, find a topic among them, create a mind map based on that topic, and provide relationship information in the form of add(node1, node2).
            However, parent-child relationships should not be duplicated.
            The words are as follows:
            체중 감량, 식이 계획, 식사 제한, 포션 컨트롤, 대체 식사, 건강한 간식, 건강한 재료, 다이어트 계획, 탄수화물, 단백질, 지방, 칼로리, 영양소, 식사 기록, 식품 레시피, 운동, 운동 루틴, 근력 훈련, 유산소 운동, 스트레스 관리, 수면, 건강한 생활, 건강 검진, 체질량 지수 (BMI), 체지방 비율, 체중 측정.
            Please answer in Korean

    """, role="user"),
    Message("""
            add("건강한 생활", "식사 관리")
            add("식사 관리", "체중 감량")
            add("식사 관리", "식이 계획")
            add("식사 관리", "식사 제한")
            add("식사 관리", "포션 컨트롤")
            add("식사 관리", "대체 식사")
            add("식사 관리", "건강한 간식")
            add("식사 관리", "건강한 재료")
            add("식사 관리", "다이어트 계획")
            add("식사 관리", "탄수화물")
            add("식사 관리", "단백질")
            add("식사 관리", "지방")
            add("식사 관리", "칼로리")
            add("식사 관리", "영양소")
            add("식사 관리", "식사 기록")
            add("식사 관리", "식품 레시피")
            add("건강한 생활", "운동 및 피트니스")
            add("운동 및 피트니스", "운동")
            add("운동 및 피트니스", "운동 루틴")
            add("운동 및 피트니스", "근력 훈련")
            add("운동 및 피트니스", "유산소 운동")
            add("건강한 생활", "건강한 습관")
            add("건강한 습관", "스트레스 관리")
            add("건강한 습관", "수면")
            add("건강한 습관", "건강 검진")
            add("건강한 생활", "체중 및 체성분 관리")
            add("체중 및 체성분 관리", "체질량 지수 (BMI)")
            add("체중 및 체성분 관리", "체지방 비율")
            add("체중 및 체성분 관리", "체중 측정")
    """, role="assistant"),
    Message("""
        You can also expand the mind map for the request. add new edges to new nodes, starting from the node "건강한 간식"
    """, role="user"),
    Message("""
        add("건강한 간식", "영양 밀도")
        add("영양 밀도", "비타민과 미네랄")
        add("영양 밀도", "섬유질")
        add("건강한 간식", "간편성")
        add("간편성", "포장 간식")
        add("간편성", "집에서 만들 수 있는 간식")
        add("건강한 간식", "맛과 다양성")
        add("맛과 다양성", "과일과 견과류")
        add("맛과 다양성", "저칼로리 디저트")
    """, role="assistant")
]



def ask_chatgpt(conversation: List[Message]) -> Tuple[str, List[Message]]:
    prompt_text = "\n".join([f"{c.role}: {c.content}" for c in conversation])

    completion = openai.Completion.create(
        model="gpt-3.5-turbo-1106",
        prompt=prompt_text,
        max_tokens=300,
        stop=None  # 필요한 경우 종료 토큰 설정
    )
    new_message_content = completion.choices[0].text.strip()
    new_message = Message(content=new_message_content, role="assistant")

    # 새 대화 내용 반환
    return new_message.content, conversation + [new_message]

class MindMap:
    """A class that represents a mind map as a graph.
    """
    
    def __init__(self, edges: Optional[List[Tuple[str, str]]]=None, nodes: Optional[List[str]]=None) -> None:
        self.edges = [] if edges is None else edges
        self.nodes = [] if nodes is None else nodes
        self.save()

    @classmethod
    def load(cls) -> MindMap:
        """Load mindmap from session state if it exists
        
        Returns: Mindmap
        """
        if "mindmap" in st.session_state:
            return st.session_state["mindmap"]
        return cls()

    def save(self) -> None:
        # save to session state
        st.session_state["mindmap"] = self

    def is_empty(self) -> bool:
        return len(self.edges) == 0
    
    def ask_for_initial_graph(self, query: str) -> None:
        """Ask GPT-3 to construct a graph from scrach.

        Args:
            query (str): The query to ask GPT-3 about.

        Returns:
            str: The output from GPT-3.
        """

        conversation = START_CONVERSATION + [
            Message(f"""
                Great, now ignore all previous nodes and restart from scratch. I now want you do the following:    

                {query}
            """, role="user")
        ]
        output, self.conversation = ask_chatgpt(conversation)
        # replace=True to restart
        me = self.parse_and_include_edges(output, replace=True)

    def ask_for_extended_graph(self, selected_node: Optional[str]=None, text: Optional[str]=None) -> None:
        # do nothing
        if (selected_node is None and text is None):
            return
        if selected_node is not None:
            conversation = self.conversation + [
                Message(f"""
                    You can also expand the mind map for the selected node. add new edges to new nodes, starting from the node "{selected_node}"
                """, role="user")
            ]
            st.session_state.last_expanded = selected_node
        else:
            # just provide the description
            conversation = self.conversation + [Message(text, role="user")]

        # now self.conversation is updated
        output, self.conversation = ask_chatgpt(conversation)
        self.parse_and_include_edges(output, replace=False)

    def parse_and_include_edges(self, output: str, replace: bool=True) -> None:
        """Parse output from LLM (GPT-3) and include the edges in the graph.

        Args:
            output (str): output from LLM (GPT-3) to be parsed
            replace (bool, optional): if True, replace all edges with the new ones, 
                otherwise add to existing edges. Defaults to True.
        """

        # Regex patterns
        pattern1 = r'(add|delete)\("([^()"]+)",\s*"([^()"]+)"\)'
        pattern2 = r'(delete)\("([^()"]+)"\)'

        # Find all matches in the text
        matches = re.findall(pattern1, output) + re.findall(pattern2, output)

        new_edges = []
        remove_edges = set()
        remove_nodes = set()
        for match in matches:
            op, *args = match
            add = op == "add"
            if add or (op == "delete" and len(args)==2):
                a, b = args
                if a == b:
                    continue
                if add:
                    new_edges.append((a, b))
                else:
                    # remove both directions
                    # (undirected graph)
                    remove_edges.add(frozenset([a, b]))
            else: # must be delete of node
                remove_nodes.add(args[0])

        if replace:
            edges = new_edges
        else:
            edges = self.edges + new_edges

        # make sure edges aren't added twice
        # and remove nodes/edges that were deleted
        added = set()
        for edge in edges:
            nodes = frozenset(edge)
            if nodes in added or nodes & remove_nodes or nodes in remove_edges:
                continue
            added.add(nodes)

        self.edges = list([tuple(a) for a in added])
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.save()

    def _delete_node(self, node) -> None:
        """Delete a node and all edges connected to it.

        Args:
            node (str): The node to delete.
        """
        self.edges = [e for e in self.edges if node not in frozenset(e)]
        self.nodes = list(set([n for e in self.edges for n in e]))
        self.conversation.append(Message(
            f'delete("{node}")', 
            role="user"
        ))
        self.save()

    def _add_expand_delete_buttons(self, node) -> None:
        st.sidebar.subheader(node)
        cols = st.sidebar.columns(2)
        cols[0].button(
            label="Expand", 
            on_click=self.ask_for_extended_graph,
            key=f"expand_{node}",
            # pass to on_click (self.ask_for_extended_graph)
            kwargs={"selected_node": node}
        )
        cols[1].button(
            label="Delete", 
            on_click=self._delete_node,
            type="primary",
            key=f"delete_{node}",
            # pass on to _delete_node
            args=(node,)
        )

    def visualize(self, graph_type: Literal["agraph", "graphviz"]) -> None:
        """Visualize the mindmap as a graph a certain way depending on the `graph_type`.

        Args:
            graph_type (Literal["agraph", "graphviz"]): The graph type to visualize the mindmap as.
        Returns:
            Union[str, None]: Any output from the clicking the graph or 
                if selecting a node in the sidebar.
        """

        selected = st.session_state.get("last_expanded")
        if graph_type == "agraph":
            vis_nodes = [
                Node(
                    id=n, 
                    label=n, 
                    # a little bit bigger if selected
                    size=10+10*(n==selected), 
                    # a different color if selected
                    color=COLOR if n != selected else FOCUS_COLOR
                ) 
                for n in self.nodes
            ]
            vis_edges = [Edge(source=a, target=b) for a, b in self.edges]
            config = Config(width="100%",
                            height=600,
                            directed=False, 
                            physics=True,
                            hierarchical=False,
                            )
            # returns a node if clicked, otherwise None
            clicked_node = agraph(nodes=vis_nodes, 
                            edges=vis_edges, 
                            config=config)
            # if clicked, update the sidebar with a button to create it
            if clicked_node is not None:
                self._add_expand_delete_buttons(clicked_node)
            return
        else: # graph_type == "graphviz":
            graph = graphviz.Graph()
            graph.attr(rankdir='LR')
            for a, b in self.edges:
                graph.edge(a, b, dir="both")
            for n in self.nodes:
                graph.node(n, style="filled", fillcolor=FOCUS_COLOR if n == selected else COLOR)
            #st.graphviz_chart(graph, use_container_width=True)
            b64 = base64.b64encode(graph.pipe(format='svg')).decode("utf-8")
            html = f"<img style='width: 100%' src='data:image/svg+xml;base64,{b64}'/>"
            st.write(html, unsafe_allow_html=True)
        # sort alphabetically
        for node in sorted(self.nodes):
            self._add_expand_delete_buttons(node)

    # will initialize the graph from session state
    # (if it exists) otherwise will create a new one


def main():
    # will initialize the graph from session state
    # (if it exists) otherwise will create a new one
    mindmap = MindMap.load()

    st.sidebar.title("영감과 창의성")

    graph_type = st.sidebar.radio("Type of graph", options=["agraph", "graphviz"])
    
    empty = mindmap.is_empty()
    reset = empty
    query = st.sidebar.text_area(
    "MINDMAP-NODE🫧에서 분석을 확인해보세요😊" if reset else "Your mind map words", 
    value=predefined_text,
    key="mindmap-input",
    height=200
    )
    submit = st.sidebar.button("마인드맵 제작하기")

    valid_submission = submit and query != ""

    if empty and not valid_submission:
        return
#어떤 작업이 진행 중
    with st.spinner(text="Loading graph..."):
        # if submit and non-empty query, then update graph
        if valid_submission:
            if reset:
                # completely new mindmap
                mindmap.ask_for_initial_graph(query=predefined_text)
            else:
                # extend existing mindmap
                mindmap.ask_for_extended_graph(text=query)
            # since inputs also have to be updated, everything
            # is rerun
            st.rerun()
        else:
            mindmap.visualize(graph_type)

# Streamlit 페이지에 제목과 날짜/시간을 추가


markdown_content = render_markdown(data)
st.set_page_config(page_title="markmap", layout="wide")


st.header('🫧Thinkwide Project Mind Map')
category = pills(
    "Category",
    list(CATEGORY_NAMES.keys()),
    CATEGORY_ICONS,
    index=None,
    format_func=lambda x: CATEGORY_NAMES.get(x, x),
    label_visibility="collapsed",
)

st.markdown("""---""")
col1, col2= st.columns(2)

with col1:
    st.subheader('🪐ThinkWide')
    st.write('**일시 :** 2022년 11월 10일 오후 6:50')
    st.write('**유형 :** 주간 회의')
    st.write('**참석자 :** 🧙🏼‍♀️봉봉님, 👨🏼‍🌾춘식님, 👨🏼‍🚒춘순님, 🕵️‍♀️춘봉님')
    st.write('**회의 요약:** ThinkWide의 VR 마인드맵 도구 개발에 초점을 맞추어, 다양한 VR 헤드셋 호환성, 가상 공간 설계의 최적화, 그리고 마케팅 전략 수립에 대해 토론하고 구체적인 작업 방향을 설정했어요!:sparkles:')

with col2:
    st.subheader('🫧Mind Map')
    with open(r'C:\CODE\thinkwide_app-main\data\structured_markdown_data1.md', encoding='utf-8') as fp:
        md = fp.read()
    markmap(md,height=250)

tab1, tab2, tab3 = st.tabs(["MINDMAP-NODE🫧", "MarkDown📊", "회의록 확인하기💫"])

with tab1:
    st.markdown('AI 추가 확장을 통해 thinkwide를 경험해보세요!')
    main()

with tab2:
    st.markdown('')
    st.markdown(markdown_content, unsafe_allow_html=True)

with tab3:
    st.markdown(f'#### ThinkWide project 회의록')
    st.markdown(meeting_data)


