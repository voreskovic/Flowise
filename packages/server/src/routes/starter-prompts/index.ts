import express from 'express'
import starterPromptsController from '../../controllers/starter-prompts'
const router = express.Router()

// POST because the generation triggers an LLM call (side-effect), not a simple read.
// The chatflowId is used to fetch the chatflow's flowData (for context) and
// its followUpPrompts config (to reuse the same LLM provider/credentials).
router.post('/:chatflowId/generate', starterPromptsController.generateStarterPrompts)

export default router
