import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.chains import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


st.set_page_config(
    page_title="LangChain - Text Summarization from YT or Website",
    page_icon="🔎"
)

st.title("🔎 LangChain - Text Summarization from YT or Website")
st.subheader('Summarize URL')


# function to extract youtube id
def get_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    else:
        return parse_qs(urlparse(url).query)["v"][0]


# GROQ api key and url input
with st.sidebar:
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password"
    )
    generic_url = st.text_input(
        "URL",
        label_visibility="collapsed"
    )


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)


prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""


prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


if st.button("Summarize the content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error(
            "Please provide a valid URL. It can be a YT video url or a website url"
        )

    else:
        try:
            with st.spinner("Waiting..."):

                # loading website or youtube data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    video_id = get_video_id(generic_url)
                    ytt_api = YouTubeTranscriptApi()
                    transcript = ytt_api.fetch(video_id)
                    text = " ".join([i.text for i in transcript])
                    data = [
                        Document(page_content=text)
                    ]

                else:

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent":
                            "Mozilla/5.0"
                        }
                    )

                    data = loader.load()


                # Chain for summarization
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )


                output_summary = chain.run(data)

                st.success(output_summary)


        except Exception as e:
            st.exception(e)