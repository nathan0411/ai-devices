import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { ChatGroq } from "@langchain/groq";
import Groq from 'groq-sdk';
import { config } from '../config';
import { traceable } from "langsmith/traceable";

const groq = new Groq();

export const generateChatCompletion = traceable(async (responseText: string) => {
    let completion;
    if (config.inferenceModelProvider === 'openai') {
        const chat = new ChatOpenAI({
            model: config.inferenceModel,
            maxTokens: 1024,
        });
        // const message = new HumanMessage(responseText);
        const messages = [
            new SystemMessage("Bạn là Robi, mascot của trường mẫu giáo Evergrin Academy. Bạn trả lời trong tối đa là 10 câu."),
            new HumanMessage(responseText),
        ];
        completion = await chat.invoke(messages);
        responseText = completion?.lc_kwargs?.content || "No information available.";
    } else if (config.inferenceModelProvider === 'groq') {
        completion = await groq.chat.completions.create({
            messages: [
                { role: "system", content: "Bạn là Robi, mascot của Evergrin Academy, trường mẫu giáo dành cho các bạn nhỏ. Bạn trả lời ngắn gọn, súc tích. Nếu bạn không biết câu trả lời, hãy nói lại câu hỏi mà sẽ được chuyển cho model tiếp theo. Bạn trả lời bằng Tiếng Việt." },
                { role: "user", content: responseText },
            ],
            model: config.inferenceModel,
        });
        responseText = completion.choices[0].message.content;
    } else {
        throw new Error('Invalid inference model provider');
    }
    return responseText;
}, { name: 'generateChatCompletion' });