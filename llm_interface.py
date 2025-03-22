import logging
import re
from typing import List, Dict, Any
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from vector_store import VectorStore
from data_preprocessing import DataPreprocessor



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, vector_store: VectorStore, df: Any, model_name: str = "EleutherAI/gpt-neo-2.7B", max_new_tokens: int = 100):
        self.vector_store = vector_store
        self.df = df  
        self.max_new_tokens = max_new_tokens
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto"
            )
            self.llm = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt"
            )
            logger.info(f"Initialized LLM with model '{model_name}' on {self.model.device}.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise

    def _build_prompt(self, question: str, context_data: List[Dict[str, Any]]) -> str:
        if not context_data:
            return f"Question: {question}\nAnswer: No relevant data found for this question"
        context = "\n".join([item['document'] for item in context_data])
        prompt = (
            f"Question: {question}\n"
            f"Context:\n{context}\n"
            f"Instructions: Provide a concise answer based on the context. "
            f"If the exact answer isn't in the context, use available data to estimate or state limitations.\n"
            f"Answer: "
        )
        return prompt

    def _post_process_response(self, response: str, question: str) -> str:
        answer_start = response.find("Answer:") + 7
        if answer_start <= 6:
            logger.warning(f"Could not parse answer from response: {response}")
            return "Failed to parse response"
        answer = response[answer_start:].strip()
        if len(answer.split()) > 100:
            answer = " ".join(answer.split()[:100]) + "..."
        return answer

    def _extract_date(self, question: str) -> str:
        month_year_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})"
        year_month_pattern = r"(\d{4}-\d{2})"
        
        month_map = {
            "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
        }
        
        match = re.search(month_year_pattern, question.lower())
        if match:
            month, year = match.groups()
            return f"{year}-{month_map[month]}"
        
        match = re.search(year_month_pattern, question)
        if match:
            return match.group(1)
        
        return None

    def generate_response(self, question: str, k: int = 5) -> str:
        if "average price" in question.lower():
            target_date = self._extract_date(question)
            if target_date:
                month_data = self.df[self.df['year_month'] == target_date]
                if not month_data.empty:
                    avg_adr = month_data['adr'].mean()
                    return f"The average daily rate for {target_date} is approximately {avg_adr:.2f}."
                else:
                    return f"No data found for average price in {target_date}."
            else:
                if not self.df.empty:
                    avg_adr = self.df['adr'].mean()
                    return f"The overall average daily rate is approximately {avg_adr:.2f}."
                else:
                    return "No data found for average price."

        if "highest booking cancellations" in question.lower():
           
            canceled_data = self.df[self.df['is_canceled'] == 1]
            if not canceled_data.empty:
                country_counts = canceled_data['country'].value_counts()
                top_country = country_counts.idxmax()
                return f"{top_country} had the highest booking cancellations."
            else:
                return "No cancellation data found."

        context_data = self.vector_store.query(question, k=k)
        prompt = self._build_prompt(question, context_data)
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_length = input_ids.input_ids.shape[-1]
        logger.info(f"Input prompt length: {input_length} tokens")
        
        try:
            response = self.llm(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]['generated_text']
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error: {str(e)}"
        
        answer = self._post_process_response(response, question)
        logger.info(f"Generated response for '{question}': {answer[:50]}.....")
        return answer

if __name__ == "__main__":
    preprocessor = DataPreprocessor("data/hotel_bookings.csv")
    df = preprocessor.get_processed_data()
    vector_store = VectorStore(df=df)
    llm = LLMInterface(vector_store, df) 
    for q in [
        "What is the average price of a hotel booking in July 2017?",
        "What is the average price of a hotel booking in March 2016?",
        "What is the average price of a hotel booking?",
        "Which locations had the highest booking cancellations?",
    ]:
        answer = llm.generate_response(q)
        logger.info(f"Q: {q}, Answer: {answer}")
