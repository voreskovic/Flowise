import { FollowUpPromptConfig, FollowUpPromptProvider, ICommonObject } from './Interface'
import { getCredentialData } from './utils'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatGoogleGenerativeAI } from '@langchain/google-genai'
import { ChatMistralAI } from '@langchain/mistralai'
import { ChatOpenAI, AzureChatOpenAI } from '@langchain/openai'
import { z } from 'zod/v3'
import { StructuredOutputParser } from '@langchain/core/output_parsers'
import { ChatGroq } from '@langchain/groq'
import { Ollama } from 'ollama'

const FollowUpPromptType = z
    .object({
        questions: z.array(z.string())
    })
    .describe('Generate Follow Up Prompts')

export interface FollowUpPromptResult {
    questions: string[]
}

/**
 * Extract sentences from source documents that were NOT already covered in the conversation.
 * Compares each sentence against the full chat history using word overlap.
 * Returns a compact string of unused content, capped to limit token cost.
 */
function extractUnusedContent(rawSourceDocuments: string, conversationText: string, maxChars: number = 1000): string {
    if (!rawSourceDocuments) return ''
    let docs: any[]
    try {
        docs = JSON.parse(rawSourceDocuments)
    } catch {
        return ''
    }
    if (!Array.isArray(docs) || docs.length === 0) return ''

    // Build a set of normalized words from the full conversation history
    const normalize = (text: string) =>
        text
            .toLowerCase()
            .replace(/[^a-z0-9\s]/g, '')
            .split(/\s+/)
            .filter((w) => w.length > 3)
    const conversationWords = new Set(normalize(conversationText))

    // Split source docs into sentences, score each against conversation
    const unusedSentences: string[] = []
    for (const doc of docs) {
        const content = doc.pageContent || ''
        const sentences = content.split(/(?<=[.!?])\s+/).filter((s: string) => s.trim().length >= 20)

        for (const sentence of sentences) {
            const words = normalize(sentence)
            if (words.length === 0) continue
            const overlapCount = words.filter((w: string) => conversationWords.has(w)).length
            const overlapRatio = overlapCount / words.length
            if (overlapRatio < 0.5) {
                unusedSentences.push(sentence.trim())
            }
        }
    }

    if (unusedSentences.length === 0) return ''

    // Cap total output to maxChars
    let result = ''
    for (const sentence of unusedSentences) {
        if (result.length + sentence.length + 2 > maxChars) break
        result += (result ? '\n' : '') + '- ' + sentence
    }
    return result
}

export const generateFollowUpPrompts = async (
    followUpPromptsConfig: FollowUpPromptConfig,
    apiMessageContent: string,
    options: ICommonObject
): Promise<FollowUpPromptResult | undefined> => {
    if (followUpPromptsConfig) {
        if (!followUpPromptsConfig.status) return undefined
        const providerConfig = followUpPromptsConfig[followUpPromptsConfig.selectedProvider]
        if (!providerConfig) return undefined
        const credentialId = providerConfig.credentialId as string
        const credentialData = await getCredentialData(credentialId ?? '', options)

        const question = (options.question as string) || ''
        const chatHistory = (options.chatHistory as string) || ''
        const rawSourceDocuments = (options.sourceDocuments as string) || ''
        const sourceProcessing = (followUpPromptsConfig as any).sourceProcessing || 'smart'

        let sources = ''
        if (sourceProcessing === 'full') {
            // Pass raw pageContent from source documents, no filtering
            try {
                const docs = JSON.parse(rawSourceDocuments)
                if (Array.isArray(docs)) {
                    sources = docs.map((doc: any) => doc.pageContent || '').join('\n\n')
                }
            } catch {
                sources = ''
            }
        } else {
            // Smart: filter out sentences already covered in conversation
            const fullConversation = chatHistory + '\n' + question + '\n' + apiMessageContent
            sources = extractUnusedContent(rawSourceDocuments, fullConversation)
        }

        const followUpPromptsPrompt = providerConfig.prompt
            .replace('{history}', apiMessageContent)
            .replace('{question}', question)
            .replace('{sources}', sources)

        switch (followUpPromptsConfig.selectedProvider) {
            case FollowUpPromptProvider.ANTHROPIC: {
                const llm = new ChatAnthropic({
                    apiKey: credentialData.anthropicApiKey,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`)
                })
                // @ts-ignore
                const structuredLLM = llm.withStructuredOutput(FollowUpPromptType, {
                    method: 'functionCalling'
                })
                const structuredResponse = await structuredLLM.invoke(followUpPromptsPrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.AZURE_OPENAI: {
                const azureOpenAIApiKey = credentialData['azureOpenAIApiKey']
                const azureOpenAIApiInstanceName = credentialData['azureOpenAIApiInstanceName']
                const azureOpenAIApiDeploymentName = credentialData['azureOpenAIApiDeploymentName']
                const azureOpenAIApiVersion = credentialData['azureOpenAIApiVersion']

                const llm = new AzureChatOpenAI({
                    azureOpenAIApiKey,
                    azureOpenAIApiInstanceName,
                    azureOpenAIApiDeploymentName,
                    azureOpenAIApiVersion,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`)
                })
                // use structured output parser because withStructuredOutput is not working
                const parser = StructuredOutputParser.fromZodSchema(FollowUpPromptType as any)
                const formatInstructions = parser.getFormatInstructions()
                const azurePrompt = followUpPromptsPrompt + '\n\n' + formatInstructions
                const chain = llm.pipe(parser)
                const structuredResponse = await chain.invoke(azurePrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.GOOGLE_GENAI: {
                const model = new ChatGoogleGenerativeAI({
                    apiKey: credentialData.googleGenerativeAPIKey,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`)
                })
                const structuredLLM = model.withStructuredOutput(FollowUpPromptType, {
                    method: 'functionCalling'
                })
                const structuredResponse = await structuredLLM.invoke(followUpPromptsPrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.MISTRALAI: {
                const model = new ChatMistralAI({
                    apiKey: credentialData.mistralAIAPIKey,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`)
                })
                // @ts-ignore
                const structuredLLM = model.withStructuredOutput(FollowUpPromptType, {
                    method: 'functionCalling'
                })
                const structuredResponse = await structuredLLM.invoke(followUpPromptsPrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.OPENAI: {
                const model = new ChatOpenAI({
                    apiKey: credentialData.openAIApiKey,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`),
                    useResponsesApi: true
                })
                // @ts-ignore
                const structuredLLM = model.withStructuredOutput(FollowUpPromptType, {
                    method: 'functionCalling'
                })
                const structuredResponse = await structuredLLM.invoke(followUpPromptsPrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.GROQ: {
                const llm = new ChatGroq({
                    apiKey: credentialData.groqApiKey,
                    model: providerConfig.modelName,
                    temperature: parseFloat(`${providerConfig.temperature}`)
                })
                const structuredLLM = llm.withStructuredOutput(FollowUpPromptType, {
                    method: 'functionCalling'
                })
                const structuredResponse = await structuredLLM.invoke(followUpPromptsPrompt)
                return structuredResponse as FollowUpPromptResult
            }
            case FollowUpPromptProvider.OLLAMA: {
                const ollamaClient = new Ollama({
                    host: providerConfig.baseUrl || 'http://127.0.0.1:11434'
                })

                const response = await ollamaClient.chat({
                    model: providerConfig.modelName,
                    messages: [
                        {
                            role: 'user',
                            content: followUpPromptsPrompt
                        }
                    ],
                    format: {
                        type: 'object',
                        properties: {
                            questions: {
                                type: 'array',
                                items: {
                                    type: 'string'
                                },
                                minItems: 3,
                                maxItems: 3,
                                description: 'Three follow-up questions based on the conversation history'
                            }
                        },
                        required: ['questions'],
                        additionalProperties: false
                    },
                    options: {
                        temperature: parseFloat(`${providerConfig.temperature}`)
                    }
                })
                const result = FollowUpPromptType.parse(JSON.parse(response.message.content))
                return result
            }
        }
    } else {
        return undefined
    }
}
