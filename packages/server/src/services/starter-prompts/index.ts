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

        // We reuse the follow-up prompts LLM provider config so the user only
        // configures credentials/model once. This is a deliberate design choice:
        // starter prompts and follow-up prompts serve similar purposes (suggesting
        // questions) and there's no reason to maintain separate LLM configurations.
        if (!chatflow.followUpPrompts) {
            throw new InternalFlowiseError(
                StatusCodes.BAD_REQUEST,
                'Follow-up Prompts must be configured with a provider before generating AI starter prompts'
            )
        }

        const followUpPromptsConfig: FollowUpPromptConfig = JSON.parse(chatflow.followUpPrompts)
        if (!followUpPromptsConfig.selectedProvider) {
            throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'No LLM provider selected in Follow-up Prompts configuration')
        }

        const context = buildContext(overrideConfig, chatflow.flowData, chatflow.name)

        // Read the user's custom prompt template from chatbotConfig, or fall back to default.
        // The user can customize this in the Starter Prompts config dialog.
        let promptTemplate =
            'Based on the following context, generate 4 short starter prompts a user might ask when first opening the chat. Each should be concise (under 100 characters), written from the user\'s perspective, and demonstrate different aspects of what this chatbot can help with.\n\nContext:\n{context}'

        if (chatflow.chatbotConfig) {
            try {
                const config = JSON.parse(chatflow.chatbotConfig)
                if (config.starterPrompts?.aiConfig?.prompt) {
                    promptTemplate = config.starterPrompts.aiConfig.prompt
                }
            } catch {
                // use default prompt
            }
        }

        // Clone the follow-up config and replace the prompt with our starter-specific one.
        // We don't mutate the original because it's still used for follow-up prompts at runtime.
        const starterConfig: FollowUpPromptConfig = JSON.parse(JSON.stringify(followUpPromptsConfig))
        const provider = starterConfig.selectedProvider
        if (starterConfig[provider]) {
            starterConfig[provider].prompt = promptTemplate.replace('{context}', context)
        }

        // Call generateFollowUpPrompts with empty history/sources/question.
        // This works because the function just formats a prompt and calls the LLM —
        // with our overridden prompt, the {history}/{question}/{sources} placeholders
        // are irrelevant since we replaced the entire prompt template above.
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
