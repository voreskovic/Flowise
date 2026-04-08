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
 * Extract character n-grams from text.
 * Handles morphological variation: "mostovi"/"mostove" share trigrams (mos,ost,sto,tov).
 * Used by the exhaustion gate for language-agnostic topic coverage detection.
 */
function extractNgrams(text: string, n: number = 3): Set<string> {
    const cleaned = text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, ' ').replace(/\s+/g, ' ').trim()
    const ngrams = new Set<string>()
    const words = cleaned.split(' ')
    for (const word of words) {
        if (word.length < n) {
            ngrams.add(word)
        } else {
            for (let i = 0; i <= word.length - n; i++) {
                ngrams.add(word.substring(i, i + n))
            }
        }
    }
    return ngrams
}

/**
 * Count how many source sentences are covered by conversation using character n-gram overlap.
 * More accurate than word overlap for morphologically rich languages (Croatian, etc.).
 */
function countUnusedSentencesNgram(
    rawSourceDocuments: string,
    conversationText: string,
    ngramSize: number = 3,
    overlapThreshold: number = 0.3
): { unusedCount: number; totalCount: number } {
    const empty = { unusedCount: 0, totalCount: 0 }
    if (!rawSourceDocuments) return empty
    let docs: any[]
    try {
        docs = JSON.parse(rawSourceDocuments)
    } catch {
        return empty
    }
    if (!Array.isArray(docs) || docs.length === 0) return empty

    const conversationNgrams = extractNgrams(conversationText, ngramSize)

    let totalCount = 0
    let unusedCount = 0
    for (const doc of docs) {
        const content = doc.pageContent || ''
        const sentences = content.split(/(?<=[.!?])\s+/).filter((s: string) => s.trim().length >= 20)
        totalCount += sentences.length

        for (const sentence of sentences) {
            const sentenceNgrams = extractNgrams(sentence, ngramSize)
            if (sentenceNgrams.size === 0) continue
            let overlapCount = 0
            for (const ng of sentenceNgrams) {
                if (conversationNgrams.has(ng)) overlapCount++
            }
            const overlapRatio = overlapCount / sentenceNgrams.size
            if (overlapRatio < overlapThreshold) {
                unusedCount++
            }
        }
    }
    return { unusedCount, totalCount }
}

interface UnusedContentResult {
    text: string
    unusedCount: number
    totalCount: number
}

/**
 * Extract sentences from source documents that were NOT already covered in the conversation.
 * Compares each sentence against the full chat history using word overlap.
 * Returns unused content text, plus sentence counts for the exhaustion gate.
 */
function extractUnusedContent(
    rawSourceDocuments: string,
    conversationText: string,
    overlapThreshold: number = 0.5,
    minWordLength: number = 3,
    maxChars: number = 1000
): UnusedContentResult {
    const empty: UnusedContentResult = { text: '', unusedCount: 0, totalCount: 0 }
    if (!rawSourceDocuments) return empty
    let docs: any[]
    try {
        docs = JSON.parse(rawSourceDocuments)
    } catch {
        return empty
    }
    if (!Array.isArray(docs) || docs.length === 0) return empty

    const conversationWords = new Set(normalize(conversationText, minWordLength))

    let totalCount = 0
    const unusedSentences: string[] = []
    for (const doc of docs) {
        const content = doc.pageContent || ''
        const sentences = content.split(/(?<=[.!?])\s+/).filter((s: string) => s.trim().length >= 20)
        totalCount += sentences.length

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

    if (unusedSentences.length === 0) return { text: '', unusedCount: 0, totalCount }

    let result = ''
    for (const sentence of unusedSentences) {
        if (result.length + sentence.length + 2 > maxChars) break
        result += (result ? '\n' : '') + '- ' + sentence
    }
    return { text: result, unusedCount: unusedSentences.length, totalCount }
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
 * Identify which previous follow-up prompts were NOT clicked by the user.
 * Compares each previous prompt against the current question using n-gram overlap.
 * Returns the unclicked prompts (candidates for carry-forward).
 */
function getUnclickedPrompts(
    previousFollowUpPrompts: string[],
    currentQuestion: string,
    ngramSize: number = 3,
    matchThreshold: number = 0.4
): string[] {
    if (!previousFollowUpPrompts.length || !currentQuestion.trim()) return previousFollowUpPrompts

    const questionNgrams = extractNgrams(currentQuestion, ngramSize)

    return previousFollowUpPrompts.filter((prompt) => {
        const promptNgrams = extractNgrams(prompt, ngramSize)
        if (promptNgrams.size === 0) return true
        let overlapCount = 0
        for (const ng of promptNgrams) {
            if (questionNgrams.has(ng)) overlapCount++
        }
        const overlapRatio = overlapCount / promptNgrams.size
        // High overlap = user clicked this one → exclude from carry-forward
        return overlapRatio < matchThreshold
    })
}

/**
 * Remove near-duplicates within a candidate list (intra-candidate dedup).
 * Keeps the first occurrence (earlier = higher priority).
 */
function deduplicateWithinCandidates(
    candidates: string[],
    threshold: number = 0.6,
    minWordLength: number = 3
): string[] {
    const result: string[] = []
    const keptSets: Set<string>[] = []

    for (const candidate of candidates) {
        const candidateWords = normalize(candidate, minWordLength)
        if (candidateWords.length === 0) {
            result.push(candidate)
            continue
        }

        let isDuplicate = false
        for (const keptSet of keptSets) {
            if (keptSet.size === 0) continue
            const overlapCount = candidateWords.filter((w) => keptSet.has(w)).length
            const overlapRatio = overlapCount / Math.max(candidateWords.length, 1)
            if (overlapRatio >= threshold) {
                isDuplicate = true
                break
            }
        }

        if (!isDuplicate) {
            result.push(candidate)
            keptSets.push(new Set(candidateWords))
        }
    }
    return result
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
        const chatHistoryMode = (followUpPromptsConfig as any).chatHistoryMode || 'all'
        const skipWhenExhausted = (followUpPromptsConfig as any).skipWhenExhausted ?? true
        const deduplicationEnabled = (followUpPromptsConfig as any).deduplicationEnabled ?? true
        const deduplicationThreshold = parseFloat(`${(followUpPromptsConfig as any).deduplicationThreshold ?? 0.6}`)

        const overlapThreshold = parseFloat(`${(followUpPromptsConfig as any).overlapThreshold ?? 0.5}`)
        const minWordLength = parseInt(`${(followUpPromptsConfig as any).minWordLength ?? 3}`, 10)
        const maxOutputChars = parseInt(`${(followUpPromptsConfig as any).maxOutputChars ?? 1000}`, 10)

        // Apply chat history windowing (for prompt context) and extract ALL previous questions (for dedup)
        const { conversationText, previousQuestions } = applyChatHistoryMode(chatHistory, question, chatHistoryMode)

        // Carry-forward: identify unclicked prompts from previous turn
        const prevFollowUps = (options.previousFollowUpPrompts as string[]) || []
        const carriedForward = getUnclickedPrompts(prevFollowUps, question)

        // Exhaustion check ALWAYS uses full unwindowed chat history — windowing only affects prompt token cost
        const fullConversationForCheck = chatHistory + '\n' + question + '\n' + apiMessageContent
        const windowedConversation = conversationText + '\n' + question + '\n' + apiMessageContent

        // Gate: deterministic exhaustion check using character n-gram overlap (handles morphological variation)
        let exhausted = false
        if (skipWhenExhausted) {
            const ngramCheck = countUnusedSentencesNgram(rawSourceDocuments, fullConversationForCheck)
            if (ngramCheck.unusedCount < 3) exhausted = true
        }

        // If exhausted but we have carried-forward prompts, return those (no LLM call)
        if (exhausted) {
            if (carriedForward.length > 0) {
                let carried = deduplicateQuestions(carriedForward, previousQuestions, deduplicationThreshold, minWordLength)
                carried = deduplicateWithinCandidates(carried, deduplicationThreshold, minWordLength)
                return carried.length > 0 ? { questions: carried.slice(0, 3) } : undefined
            }
            return undefined
        }

        let sources = ''
        if (sourceProcessing === 'full') {
            try {
                const docs = JSON.parse(rawSourceDocuments)
                if (Array.isArray(docs)) {
                    sources = docs.map((doc: any) => doc.pageContent || '').join('\n\n')
                }
            } catch {
                sources = ''
            }
        } else {
            const result = extractUnusedContent(rawSourceDocuments, windowedConversation, overlapThreshold, minWordLength, maxOutputChars)
            sources = result.text
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

        // Merge: fresh LLM results first (priority), then carried-forward
        let merged: string[] = [...(llmResult?.questions || []), ...carriedForward]

        // Dedup against previously asked questions
        if (deduplicationEnabled && previousQuestions.length > 0) {
            merged = deduplicateQuestions(merged, previousQuestions, deduplicationThreshold, minWordLength)
        }

        // Dedup within the merged list (removes near-duplicate candidates)
        merged = deduplicateWithinCandidates(merged, deduplicationThreshold, minWordLength)

        // Take top 3
        merged = merged.slice(0, 3)

        return merged.length > 0 ? { questions: merged } : undefined
    } else {
        return undefined
    }
}
