from rag_helper import RAGBase

class RAGHomework1(RAGBase):

    def search(self, query: str) -> dict[str, str]:
        """
        Search the chuncked github md files using the index and return the top 5 results as a list of dicts with keys "filename" and "content".
        """
        return self.index.search(
            query,
            num_results=5,
            boost_dict={"content": 3.0},
            filter_dict={}
        )


    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()
    
    #returnst the entire response and not just the output text, which we need for the homework
    def llm_full_response(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response
    
    #returns the entire response, not just the output text
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm_full_response(prompt)
        return answer