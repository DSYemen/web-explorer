from langchain import WebResearchRetriever, RetrieverResult
from faiss import IndexFlatL2
from streamlit import Streamlit

# تعريف قاعدة بيانات FAISS
index = IndexFlatL2()

# تعريف مسترجع WebResearchRetriever
retriever = WebResearchRetriever()

# دالة للبحث عن الأسئلة
def search(question):
    # استخراج تضمين السؤال
    question_embedding = retriever.get_question_embedding(question)

    # البحث عن أقرب 10 نتائج
    results = index.search(question_embedding, 10)

    # تحويل نتائج المسترجع إلى قائمة
    retriever_results = [
        RetrieverResult(
            title=result[0],
            url=result[1],
            score=result[2],
        )
        for result in results
    ]

    return retriever_results

# واجهة Streamlit
streamlit = Streamlit()

# نموذج إدخال السؤال
question = streamlit.text_input("أدخل سؤالك هنا")

# زر البحث
if streamlit.button("بحث"):
    # عرض نتائج البحث
    results = search(question)
    for result in results:
        streamlit.write(f"**العنوان:** {result.title}")
        streamlit.write(f"**الرابط:** {result.url}")
        streamlit.write(f"**الدرجة:** {result.score}")
        streamlit.write("---")

