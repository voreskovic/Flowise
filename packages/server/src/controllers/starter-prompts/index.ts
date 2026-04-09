import { Request, Response, NextFunction } from 'express'
import starterPromptsService from '../../services/starter-prompts'

const generateStarterPrompts = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const chatflowId = req.params.chatflowId
        if (!chatflowId) {
            throw new Error('chatflowId is required')
        }
        // overrideConfig comes from the mobile/embed client and contains
        // vars.story_content (the main topic text) and qdrantFilter (document IDs).
        // This is the same payload the client sends on first prediction,
        // but now sent at chatbot open time so we can generate prompts early.
        const overrideConfig = req.body.overrideConfig || {}
        const apiResponse = await starterPromptsService.generateStarterPrompts(chatflowId, overrideConfig)
        return res.json(apiResponse)
    } catch (error) {
        next(error)
    }
}

export default {
    generateStarterPrompts
}
