import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'
import { getErrorMessage } from '../../errors/utils'
import { getRunningExpressApp } from '../../utils/getRunningExpressApp'
import { databaseEntities } from '../../utils'
import { ChatFlow } from '../../database/entities/ChatFlow'
import { generateFollowUpPrompts, FollowUpPromptConfig } from 'flowise-components'

/**
 * Build the LLM context string from the mobile client's overrideConfig.
 *
 * The primary context source is `vars.story_content` — this is the full topic/article
 * text that the chatbot session is about. We also include any other string vars
 * (like cluster_id) for additional context, and append node system messages from
 * the chatflow's flowData so the LLM understands the chatbot's persona.
 */
function buildContext(overrideConfig: Record<string, any>, flowData: string, chatflowName: string): string {
    const parts: string[] = []

    // Primary context: story_content from vars (the main topic the user will ask about)
    const vars = overrideConfig.vars || {}
    if (vars.story_content) {
        parts.push(`Topic content:\n${vars.story_content}`)
    }

    // Secondary context: chatflow name + system messages from flow nodes
    // This helps the LLM understand the chatbot's role/persona
    parts.push(`Chatflow: ${chatflowName}`)
    try {
        const flow = JSON.parse(flowData)
        if (flow.nodes && Array.isArray(flow.nodes)) {
            for (const node of flow.nodes) {
                const inputs = node.data?.inputs || {}
                for (const key of ['systemMessage', 'systemMessagePrompt', 'instructions']) {
                    if (inputs[key] && typeof inputs[key] === 'string' && inputs[key].trim()) {
                        parts.push(`System message: ${inputs[key].trim()}`)
                        break // one system message is enough context
                    }
                }
            }
        }
    } catch {
        // flowData parsing failed — continue with what we have
    }

    const result = parts.join('\n\n')
    return result.length > 3000 ? result.slice(0, 3000) : result
}

const generateStarterPrompts = async (chatflowId: string, overrideConfig: Record<string, any>) => {
    try {
        const appServer = getRunningExpressApp()
        const chatflow = await appServer.AppDataSource.getRepository(ChatFlow).findOneBy({ id: chatflowId })
        if (!chatflow) {
            throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Chatflow ${chatflowId} not found`)
        }

        // Read the AI config from chatbotConfig.starterPrompts.aiConfig
        // This has its own provider/credential/model — independent from follow-up prompts.
        if (!chatflow.chatbotConfig) {
            throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Chatbot config is not set')
        }

        let starterAiConfig: any
        try {
            const config = JSON.parse(chatflow.chatbotConfig)
            starterAiConfig = config.starterPrompts?.aiConfig
        } catch {
            throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Failed to parse chatbot config')
        }

        if (!starterAiConfig || !starterAiConfig.selectedProvider) {
            throw new InternalFlowiseError(
                StatusCodes.BAD_REQUEST,
                'AI Starter Prompts must be configured with a provider before generating'
            )
        }

        const context = buildContext(overrideConfig, chatflow.flowData, chatflow.name)

        // Read the prompt template from the provider config, or fall back to default.
        const provider = starterAiConfig.selectedProvider
        const providerConfig = starterAiConfig[provider] || {}
        let promptTemplate =
            providerConfig.prompt ||
            'Based on the following context, generate 4 short starter prompts a user might ask when first opening the chat. Each should be concise (under 100 characters), written from the user\'s perspective, and demonstrate different aspects of what this chatbot can help with.\n\nContext:\n{context}'

        // Build the FollowUpPromptConfig shape so we can reuse generateFollowUpPrompts.
        // Force status=true and skipWhenExhausted=false since there's no conversation to exhaust.
        const starterConfig: FollowUpPromptConfig = {
            status: true,
            selectedProvider: provider,
            [provider]: {
                ...providerConfig,
                prompt: promptTemplate.replace('{context}', context)
            },
            skipWhenExhausted: false,
            deduplicationEnabled: false
        } as any

        const result = await generateFollowUpPrompts(starterConfig, '', {
            chatId: '',
            chatflowid: chatflowId,
            appDataSource: appServer.AppDataSource,
            databaseEntities: databaseEntities,
            question: '',
            sourceDocuments: '',
            chatHistory: '',
            analytic: chatflow.analytic || ''
        })

        return result || { questions: [] }
    } catch (error) {
        if (error instanceof InternalFlowiseError) throw error
        throw new InternalFlowiseError(StatusCodes.INTERNAL_SERVER_ERROR, `Error generating starter prompts: ${getErrorMessage(error)}`)
    }
}

export default {
    generateStarterPrompts
}
