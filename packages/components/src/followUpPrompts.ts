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
import CallbackHandler from 'langfuse-langchain'

const FollowUpPromptType = z
    .object({
        questions: z.array(z.string())
    })
    .describe('Generate Follow Up Prompts')

export interface FollowUpPromptResult {
    questions: string[]
}

/**
 * Normalize text into lowercase words for overlap comparison.
 * Shared by extractUnusedContent, extractUserQuestions, and deduplication logic.
 */
function normalize(text: string, minWordLength: number = 3): string[] {
    return text
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, '')
        .split(/\s+/)
        .filter((w) => w.length >= minWordLength)
}

/**
 * Extract sentences from source documents that were NOT already covered in the conversation.
 * Compares each sentence against the full chat history using word overlap.
 * Returns a compact string of unused content, capped to limit token cost.
 */
function extractUnusedContent(
    rawSourceDocuments: string,
    conversationText: string,
    overlapThreshold: number = 0.5,
    minWordLength: number = 3,
    maxChars: number = 1000
): string {
    if (!rawSourceDocuments) return ''
    let docs: any[]
    try {
        docs = JSON.parse(rawSourceDocuments)
    } catch {
        return ''
    }
    if (!Array.isArray(docs) || docs.length === 0) return ''

    const conversationWords = new Set(normalize(conversationText, minWordLength))

    const unusedSentences: string[] = []
    for (const doc of docs) {
        const content = doc.pageContent || ''
        const sentences = content.split(/(?<=[.!?])\s+/).filter((s: string) => s.trim().length >= 20)

        for (const sentence of sentences) {
            const words = normalize(sentence, minWordLength)
            if (words.length === 0) continue
            const overlapCount = words.filter((w: string) => conversationWords.has(w)).length
            const overlapRatio = overlapCount / words.length
            if (overlapRatio < overlapThreshold) {
                unusedSentences.push(sentence.trim())
            }
        }
    }

    if (unusedSentences.length === 0) return ''

    let result = ''
    for (const sentence of unusedSentences) {
        if (result.length + sentence.length + 2 > maxChars) break
        result += (result ? '\n' : '') + '- ' + sentence
    }
    return result
}

/**
 * Extract user questions from chat history text.
 * Chat history format: "Human: ...\nAssistant: ...\n"
 */
function extractUserQuestions(chatHistory: string, currentQuestion: string): string[] {
    const questions: string[] = []
    const lines = chatHistory.split('\n')
    for (const line of lines) {
        const match = line.match(/^Human:\s*(.+)$/i)
        if (match && match[1].trim().length > 0) {
            questions.push(match[1].trim())
        }
    }
    if (currentQuestion.trim()) {
        questions.push(currentQuestion.trim())
    }
    return questions
}

/**
 * Apply chat history windowing based on the configured mode.
 * Returns the processed chat history text and extracted user questions.
 */
function applyChatHistoryMode(
    chatHistory: string,
    currentQuestion: string,
    mode: string
): { conversationText: string; previousQuestions: string[] } {
    const lines = chatHistory.split('\n').filter((l) => l.trim().length > 0)

    // Parse into message pairs
    const messages: { role: string; content: string }[] = []
    for (const line of lines) {
        const humanMatch = line.match(/^Human:\s*(.+)$/i)
        const assistantMatch = line.match(/^Assistant:\s*(.+)$/i)
        if (humanMatch) messages.push({ role: 'human', content: humanMatch[1].trim() })
        else if (assistantMatch) messages.push({ role: 'assistant', content: assistantMatch[1].trim() })
    }

    let filteredMessages: { role: string; content: string }[]
    switch (mode) {
        case 'last-3':
            filteredMessages = messages.slice(-3)
            break
        case 'last-5':
            filteredMessages = messages.slice(-5)
            break
        case 'last-10':
            filteredMessages = messages.slice(-10)
            break
        case 'all':
            filteredMessages = messages
            break
        case 'user-only':
            filteredMessages = messages.filter((m) => m.role === 'human')
            break
        case 'user-and-ai':
            filteredMessages = messages
            break
        default:
            filteredMessages = messages.slice(-3)
    }

    const conversationText = filteredMessages.map((m) => m.content).join('\n')
    const previousQuestions = extractUserQuestions(chatHistory, currentQuestion)

    return { conversationText, previousQuestions }
}

/**
 * Deterministic post-filter: remove generated questions that are too similar
 * to previously asked questions using word overlap.
 */
function deduplicateQuestions(
    candidates: string[],
    previousQuestions: string[],
    threshold: number = 0.6,
    minWordLength: number = 3
): string[] {
    if (previousQuestions.length === 0) return candidates

    const prevSets = previousQuestions.map((q) => new Set(normalize(q, minWordLength)))

    return candidates.filter((candidate) => {
        const candidateWords = normalize(candidate, minWordLength)
        if (candidateWords.length === 0) return true
        for (const prevSet of prevSets) {
            if (prevSet.size === 0) continue
            const overlapCount = candidateWords.filter((w) => prevSet.has(w)).length
            const overlapRatio = overlapCount / Math.max(candidateWords.length, 1)
            if (overlapRatio >= threshold) return false
        }
        return true
    })
}

/**
 * Build LangChain callbacks for tracing follow-up prompt generation.
 * Creates a separate Langfuse trace grouped under the same session (chatId).
 */
async function buildTracingCallbacks(options: ICommonObject): Promise<any[]> {
    const callbacks: any[] = []
    if (!options.analytic) return callbacks
    try {
        const analytic = JSON.parse(options.analytic)
        if (analytic.langFuse?.status) {
            const credentialData = await getCredentialData(analytic.langFuse.credentialId ?? '', options)
            const handler = new CallbackHandler({
                secretKey: credentialData.langFuseSecretKey,
                publicKey: credentialData.langFusePublicKey,
                baseUrl: credentialData.langFuseEndpoint ?? 'https://cloud.langfuse.com',
                sdkIntegration: 'Flowise',
                sessionId: options.chatId,
                metadata: { source: 'follow-up-prompts' }
            })
            callbacks.push(handler)
        }
    } catch {
        // analytic config not available or invalid — skip tracing
    }
    return callbacks
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
        const callbacks = await buildTracingCallbacks(options)

        const question = (options.question as string) || ''
        const chatHistory = (options.chatHistory as string) || ''
        const rawSourceDocuments = (options.sourceDocuments as string) || ''
        const sourceProcessing = (followUpPromptsConfig as any).sourceProcessing || 'smart'
        const chatHistoryMode = (followUpPromptsConfig as any).chatHistoryMode || 'last-3'
        const skipWhenExhausted = (followUpPromptsConfig as any).skipWhenExhausted ?? true
        const deduplicationEnabled = (followUpPromptsConfig as any).deduplicationEnabled ?? true
        const deduplicationThreshold = parseFloat(`${(followUpPromptsConfig as any).deduplicationThreshold ?? 0.6}`)

        const overlapThreshold = parseFloat(`${(followUpPromptsConfig as any).overlapThreshold ?? 0.5}`)
        const minWordLength = parseInt(`${(followUpPromptsConfig as any).minWordLength ?? 3}`, 10)
        const maxOutputChars = parseInt(`${(followUpPromptsConfig as any).maxOutputChars ?? 1000}`, 10)

        // Apply chat history windowing and extract previous questions
        const { conversationText, previousQuestions } = applyChatHistoryMode(chatHistory, question, chatHistoryMode)
        const fullConversation = conversationText + '\n' + question + '\n' + apiMessageContent

        let sources = ''
        if (sourceProcessing === 'full') {
            // Gate: even in full mode, check if topics are exhausted before calling LLM
            if (skipWhenExhausted) {
                const unusedCheck = extractUnusedContent(rawSourceDocuments, fullConversation, overlapThreshold, minWordLength, maxOutputChars)
                if (!unusedCheck) return undefined
            }
            try {
                const docs = JSON.parse(rawSourceDocuments)
                if (Array.isArray(docs)) {
                    sources = docs.map((doc: any) => doc.pageContent || '').join('\n\n')
                }
            } catch {
                sources = ''
            }
        } else {
            sources = extractUnusedContent(rawSourceDocuments, fullConversation, overlapThreshold, minWordLength, maxOutputChars)
            // Gate: no unused content means all topics covered — skip LLM call
            if (skipWhenExhausted && !sources) return undefined
        }

        const previousQuestionsText = previousQuestions.length > 0 ? previousQuestions.map((q) => '- ' + q).join('\n') : 'None yet.'

        const followUpPromptsPrompt = providerConfig.prompt
            .replace('{history}', apiMessageContent)
            .replace('{question}', question)
            .replace('{sources}', sources)
            .replace('{previousQuestions}', previousQuestionsText)

        // Call LLM provider and collect raw result
        let llmResult: FollowUpPromptResult | undefined
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
                llmResult = (await structuredLLM.invoke(followUpPromptsPrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = (await chain.invoke(azurePrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = (await structuredLLM.invoke(followUpPromptsPrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = (await structuredLLM.invoke(followUpPromptsPrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = (await structuredLLM.invoke(followUpPromptsPrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = (await structuredLLM.invoke(followUpPromptsPrompt, { callbacks })) as FollowUpPromptResult
                break
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
                llmResult = FollowUpPromptType.parse(JSON.parse(response.message.content))
                break
            }
        }

        if (!llmResult?.questions?.length) return undefined

        // Deterministic post-filter: remove questions too similar to previously asked ones
        if (deduplicationEnabled && previousQuestions.length > 0) {
            llmResult.questions = deduplicateQuestions(llmResult.questions, previousQuestions, deduplicationThreshold, minWordLength)
        }

        return llmResult.questions.length > 0 ? llmResult : undefined
    } else {
        return undefined
    }
}
