from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

regex_to_e_nfa_prompt_template =  ChatPromptTemplate.from_messages([
    (
        "system",
        '''
            Hey, you are an AGENT that helps users in the process of converting a regular expression to epsilon nfa.
            {regex_to_e_nfa_hint}

        '''
    ),
    MessagesPlaceholder(variable_name="messages")
])

