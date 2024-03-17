import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import kusto_wingman as kw

st.set_page_config(page_title="Kusto Wingman", layout="wide")
css = '''
<style>
    [data-testid="stSidebar"]{
        min-width: 400px;
        max-width: 800px;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)

st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(http://placekitten.com/200/200);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# sidebar
prompt = st.sidebar.chat_input('message', key='send_message_chatinput')
if prompt:
    with st.chat_message(name='main-chat-msg'):
        st.markdown(prompt)

    respMsg, queryResult = kw.create_message(st.session_state.threadId, st.session_state.assistantId, prompt)

    with st.chat_message(name='main-chat-msg'):
        st.markdown(respMsg)

st.sidebar.button('Send', key='send_message_btn')
st.sidebar.text_area('Access Token', key='access_token_txtbox', height=400)

with st.container(height=350, border=True):
    st.chat_message(name='main-chat-msg')
    
with st.container(height=350, border=True):
    st.table(data={
        'subscription ID': ['aaa'],
        'Resource Group': ['bbb'],
        'Name': ['redis-001'],
        'Properties': ['{id: "/subscriptionid/..."}']
    })

threadId, assistantId = kw.init()
st.session_state.threadId = threadId
st.session_state.assistantId = assistantId

