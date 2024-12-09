import streamlit as st
from utils import load_pdf, create_vector_store, ask_question
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def main():
    # Configuração do Streamlit
    st.set_page_config(page_title='Converse com seus arquivos', page_icon=':books:') 
    st.title("🤖 Chatbot com PDF")

    # Inicializar estados para o vetor, histórico e input
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "question" not in st.session_state:
        st.session_state["question"] = ""

    with st.sidebar:
        # Carregar o arquivo PDF
        st.subheader('Seus arquivos')
        uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")
        if uploaded_file:
            st.success("PDF carregado com sucesso! Extraindo o conteúdo...")
            
            # Extrair texto do PDF
            text = load_pdf(uploaded_file)
            
            # Criar vetor de busca com FAISS
            st.session_state["vector_store"] = create_vector_store(text, OpenAIEmbeddings())

            st.success("Base de conhecimento criada! Pergunte algo ao chatbot.")

    # Verificar se o vetor de busca está pronto
    if st.session_state["vector_store"] is not None:
        # Formulário para perguntas
        with st.form(key="question_form"):
            question = st.text_input("Faça sua pergunta:", value=st.session_state["question"], key="input_question")
            submitted = st.form_submit_button("Enviar")
        
        if submitted and question:
            # Obter a resposta do chatbot
            answer = ask_question(question, st.session_state["vector_store"])
            
            # Salvar a pergunta e a resposta no histórico
            st.session_state["chat_history"].append({"question": question, "answer": answer})
            
            # Limpar a caixa de entrada
            st.session_state["question"] = ""

        # Exibir o histórico de perguntas e respostas
        for chat in st.session_state["chat_history"]:
            st.write(f"**Pergunta:** {chat['question']}")
            st.write(f"**Resposta:** {chat['answer']}")
            st.write("---")  # Linha divisória entre interações
    else:
        st.info("Envie um arquivo PDF na barra lateral para começar.")

if __name__ == "__main__":
    main()
