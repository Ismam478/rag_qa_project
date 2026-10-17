from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

# Import YOUR raw retrieval function
from src.db_retrieval import retrieve_parent_text

load_dotenv(find_dotenv(), override=True)

def main():
    open_ai_client = OpenAI()
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    questions = [
    "What are the main changes in the third edition of ISO/IEC 27001 compared to the second edition?",
    "How does the information security management system preserve the confidentiality, integrity, and availability of information?",
    "Is it acceptable to exclude any of the requirements specified in Clauses 4 to 10 when claiming conformity to ISO/IEC 27001:2022?",
    "What must an organization determine to establish the external and internal context that affects its information security management system?",
    "When understanding the needs and expectations of interested parties, what three things must an organization determine?",
    "What three factors must an organization consider when determining the scope of the information security management system?",
    "How can top management demonstrate leadership and commitment regarding the information security policy and objectives?",
    "What commitments must be included in the established information security policy?",
    "What are the two main criteria that must be established and maintained within the information security risk assessment process?",
    "What three components are involved in analyzing information security risks?",
    "What four pieces of information must be contained in the Statement of Applicability produced during information security risk treatment?",
    "According to Clause 6.2, what are five of the characteristics required for information security objectives?",
    "According to Clause 7.2, on what basis must an organization ensure that persons affecting information security performance are competent?",
    "What three things must persons doing work under the organization's control be aware of?",
    "What identification and description details should be ensured when creating and updating documented information?",
    "When must an organization perform information security risk assessments according to Clause 8.2?",
    "Why must an organization conduct internal audits at planned intervals?",
    "What should an organization consider when establishing an internal audit programme?",
    "What must top management consider regarding previous management reviews during a current management review?",
    "When a nonconformity occurs, what initial two actions must the organization take?",
    "According to Annex A, Control 5.3, what must be done regarding conflicting duties and areas of responsibility?",
    "What action is required regarding the organization's assets upon change or termination of employment?",
    "According to Control 5.12, what criteria should be used to classify information?",
    "How should an organization handle the identification, collection, acquisition, and preservation of evidence related to information security events?",
    "When should background verification checks on candidates be carried out?",
    "What must be defined and appropriately enforced regarding workspaces and screens?",
    "What verification is required prior to the disposal or re-use of items of equipment containing storage media?",
    "When should information stored in information systems, devices, or any other storage media be deleted?",
    "How should groups of information services, users, and information systems be managed within the organization's networks?",
    "According to Control 8.24, what specific rule must be defined and implemented for the effective use of cryptography?"
    ]

    ground_truths = [
    "The text has been aligned with the harmonized structure for management system standards and ISO/IEC 27002:2022[cite: 25].",
    "It preserves them by applying a risk management process and gives confidence to interested parties that risks are adequately managed[cite: 26].",
    "No, excluding any of the requirements specified in Clauses 4 to 10 is not acceptable when an organization claims conformity to this document[cite: 28].",
    "The organization shall determine external and internal issues that are relevant to its purpose and that affect its ability to achieve the intended outcome(s)[cite: 28].",
    "The organization must determine relevant interested parties, their relevant requirements, and which of these requirements will be addressed through the information security management system[cite: 28].",
    "The organization must consider external and internal issues, the requirements of interested parties, and interfaces and dependencies between activities performed by the organization and other organizations[cite: 29].",
    "By ensuring the information security policy and the information security objectives are established and are compatible with the strategic direction of the organization[cite: 29].",
    "It must include a commitment to satisfy applicable requirements related to information security and a commitment to continual improvement of the information security management system[cite: 30].",
    "The risk acceptance criteria and the criteria for performing information security risk assessments[cite: 31].",
    "Assessing the potential consequences if risks materialize, assessing the realistic likelihood of occurrence, and determining the levels of risk[cite: 31].",
    "The necessary controls, justification for their inclusion, whether the necessary controls are implemented or not, and the justification for excluding any of the Annex A controls[cite: 31, 32].",
    "They shall be consistent with the information security policy, measurable (if practicable), monitored, communicated, and updated as appropriate[cite: 32].",
    "On the basis of appropriate education, training, or experience[cite: 33].",
    "The information security policy, their contribution to the effectiveness of the information security management system, and the implications of not conforming with the requirements[cite: 33].",
    "Identification and description such as a title, date, author, or reference number[cite: 34].",
    "At planned intervals or when significant changes are proposed or occur[cite: 35].",
    "To provide information on whether the information security management system conforms to the organization's own requirements and the requirements of this document, and if it is effectively implemented and maintained[cite: 36].",
    "The organization shall consider the importance of the processes concerned and the results of previous audits[cite: 36].",
    "The status of actions from previous management reviews[cite: 37].",
    "The organization must take action to control and correct it, and deal with the consequences[cite: 37].",
    "Conflicting duties and conflicting areas of responsibility shall be segregated[cite: 38].",
    "Personnel and other interested parties shall return all the organization's assets in their possession[cite: 38].",
    "Information shall be classified according to the information security needs of the organization based on confidentiality, integrity, availability and relevant interested party requirements[cite: 39].",
    "The organization shall establish and implement procedures for these actions[cite: 40].",
    "Prior to joining the organization and on an ongoing basis taking into consideration applicable laws, regulations and ethics[cite: 41].",
    "Clear desk rules for papers and removable storage media and clear screen rules for information processing facilities[cite: 41].",
    "They shall be verified to ensure that any sensitive data and licensed software has been removed or securely overwritten[cite: 42].",
    "It shall be deleted when no longer required[cite: 43].",
    "They shall be segregated in the organization's networks[cite: 44].",
    "Rules for the effective use of cryptography must include cryptographic key management[cite: 44]."
    ]


    answers = []
    contexts = []

    print("Running pipeline against Golden Dataset...")


    for i, query in enumerate(questions):
        print(f"\nProcessing Question {i+1}/{len(questions)}: '{query}'")
        

        retrieved_texts = retrieve_parent_text(query)
        contexts.append(retrieved_texts) 
        

        context_block = "\n\n".join(retrieved_texts)
        system_prompt = (
            "You are a helpful assistant that extracts relevant information from the following document and answers questions about it:\n\n"
            f"Document: {context_block}\n\n"
            "If you don't know the answer, just say 'I don't know'. Do not try to make up an answer."
        )

        response = open_ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        answers.append(answer) 



    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("\n--------------------------------------------------")
    print("Pipeline execution complete. Handing data to Ragas for grading...")
    print("--------------------------------------------------\n")


    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,   
            faithfulness,      
            answer_relevancy   
        ],
        llm = evaluator_llm,
        embeddings = evaluator_embeddings
    )


    print("\n=== RAG SYSTEM GRADES (0.0 to 1.0) ===")
    df = result.to_pandas()

    print(result)
    
    df.to_csv("rag_evaluation_report.csv", index=False)
    print("\nDetailed breakdown saved to 'rag_evaluation_report.csv'")

if __name__ == "__main__":
    main()