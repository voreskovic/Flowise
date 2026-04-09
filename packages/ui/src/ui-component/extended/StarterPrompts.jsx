import { useDispatch } from 'react-redux'
import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'
import { enqueueSnackbar as enqueueSnackbarAction, closeSnackbar as closeSnackbarAction, SET_CHATFLOW } from '@/store/actions'

// material-ui
import { Button, IconButton, OutlinedInput, Box, List, InputAdornment, Typography, CircularProgress } from '@mui/material'
import { IconX, IconTrash, IconPlus, IconBulb, IconSparkles, IconRefresh } from '@tabler/icons-react'

// Project import
import { StyledButton } from '@/ui-component/button/StyledButton'
import { SwitchInput } from '@/ui-component/switch/Switch'
import { Input } from '@/ui-component/input/Input'

// store
import useNotifier from '@/utils/useNotifier'

// API
import chatflowsApi from '@/api/chatflows'

const defaultAiPrompt =
    'Based on the following context, generate 4 short starter prompts a user might ask when first opening the chat. Each should be concise (under 100 characters), written from the user\'s perspective, and demonstrate different aspects of what this chatbot can help with.\n\nContext:\n{context}'

const StarterPrompts = ({ dialogProps, onConfirm }) => {
    const dispatch = useDispatch()

    useNotifier()

    const enqueueSnackbar = (...args) => dispatch(enqueueSnackbarAction(...args))
    const closeSnackbar = (...args) => dispatch(closeSnackbarAction(...args))

    const [inputFields, setInputFields] = useState([
        {
            prompt: ''
        }
    ])

    const [chatbotConfig, setChatbotConfig] = useState({})
    const [aiEnabled, setAiEnabled] = useState(false)
    const [aiPrompt, setAiPrompt] = useState(defaultAiPrompt)
    const [isGenerating, setIsGenerating] = useState(false)
    const [generatedPrompts, setGeneratedPrompts] = useState([])
    const [hasFollowUpConfig, setHasFollowUpConfig] = useState(false)

    const addInputField = () => {
        setInputFields([
            ...inputFields,
            {
                prompt: ''
            }
        ])
    }
    const removeInputFields = (index) => {
        const rows = [...inputFields]
        rows.splice(index, 1)
        setInputFields(rows)
    }

    const handleChange = (index, evnt) => {
        const { name, value } = evnt.target
        const list = [...inputFields]
        list[index][name] = value
        setInputFields(list)
    }

    const handleGenerate = async () => {
        try {
            setIsGenerating(true)
            setGeneratedPrompts([])
            const resp = await chatflowsApi.generateStarterPrompts(dialogProps.chatflow.id, { overrideConfig: {} })
            if (resp.data && resp.data.questions && resp.data.questions.length > 0) {
                setGeneratedPrompts(resp.data.questions)
                // Also set them as the manual input fields so the user can edit before saving
                setInputFields(resp.data.questions.map((q) => ({ prompt: q })))
                enqueueSnackbar({
                    message: `Generated ${resp.data.questions.length} starter prompts`,
                    options: {
                        key: new Date().getTime() + Math.random(),
                        variant: 'success',
                        action: (key) => (
                            <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                                <IconX />
                            </Button>
                        )
                    }
                })
            } else {
                enqueueSnackbar({
                    message: 'No prompts were generated. Try adjusting the prompt template.',
                    options: {
                        key: new Date().getTime() + Math.random(),
                        variant: 'warning',
                        action: (key) => (
                            <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                                <IconX />
                            </Button>
                        )
                    }
                })
            }
        } catch (error) {
            const errorMsg =
                typeof error.response?.data === 'object' ? error.response.data.message : error.response?.data || error.message
            enqueueSnackbar({
                message: `Failed to generate starter prompts: ${errorMsg}`,
                options: {
                    key: new Date().getTime() + Math.random(),
                    variant: 'error',
                    persist: true,
                    action: (key) => (
                        <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                            <IconX />
                        </Button>
                    )
                }
            })
        } finally {
            setIsGenerating(false)
        }
    }

    const onSave = async () => {
        try {
            let value = {
                starterPrompts: {
                    ...inputFields,
                    aiConfig: {
                        status: aiEnabled,
                        prompt: aiPrompt
                    }
                }
            }
            chatbotConfig.starterPrompts = value.starterPrompts
            const saveResp = await chatflowsApi.updateChatflow(dialogProps.chatflow.id, {
                chatbotConfig: JSON.stringify(chatbotConfig)
            })
            if (saveResp.data) {
                enqueueSnackbar({
                    message: 'Conversation Starter Prompts Saved',
                    options: {
                        key: new Date().getTime() + Math.random(),
                        variant: 'success',
                        action: (key) => (
                            <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                                <IconX />
                            </Button>
                        )
                    }
                })
                dispatch({ type: SET_CHATFLOW, chatflow: saveResp.data })
                onConfirm?.()
            }
        } catch (error) {
            enqueueSnackbar({
                message: `Failed to save Conversation Starter Prompts: ${
                    typeof error.response.data === 'object' ? error.response.data.message : error.response.data
                }`,
                options: {
                    key: new Date().getTime() + Math.random(),
                    variant: 'error',
                    persist: true,
                    action: (key) => (
                        <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                            <IconX />
                        </Button>
                    )
                }
            })
        }
    }

    useEffect(() => {
        if (dialogProps.chatflow) {
            // Check if follow-up prompts are configured (needed for AI generation)
            if (dialogProps.chatflow.followUpPrompts) {
                try {
                    const fupConfig = JSON.parse(dialogProps.chatflow.followUpPrompts)
                    setHasFollowUpConfig(!!fupConfig.selectedProvider)
                } catch {
                    setHasFollowUpConfig(false)
                }
            }

            if (dialogProps.chatflow.chatbotConfig) {
                try {
                    let chatbotConfig = JSON.parse(dialogProps.chatflow.chatbotConfig)
                    setChatbotConfig(chatbotConfig || {})
                    if (chatbotConfig.starterPrompts) {
                        let inputFields = []
                        Object.getOwnPropertyNames(chatbotConfig.starterPrompts).forEach((key) => {
                            if (key !== 'aiConfig' && chatbotConfig.starterPrompts[key]) {
                                inputFields.push(chatbotConfig.starterPrompts[key])
                            }
                        })
                        if (inputFields.length > 0) {
                            setInputFields(inputFields)
                        }

                        // Load AI config
                        if (chatbotConfig.starterPrompts.aiConfig) {
                            setAiEnabled(chatbotConfig.starterPrompts.aiConfig.status || false)
                            if (chatbotConfig.starterPrompts.aiConfig.prompt) {
                                setAiPrompt(chatbotConfig.starterPrompts.aiConfig.prompt)
                            }
                        }
                    } else {
                        setInputFields([{ prompt: '' }])
                    }
                } catch (e) {
                    setInputFields([{ prompt: '' }])
                }
            }
        }

        return () => {}
    }, [dialogProps])

    return (
        <>
            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    borderRadius: 10,
                    background: '#d8f3dc',
                    padding: 10
                }}
            >
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'row',
                        alignItems: 'center'
                    }}
                >
                    <IconBulb size={30} color='#2d6a4f' />
                    <span style={{ color: '#2d6a4f', marginLeft: 10, fontWeight: 500 }}>
                        Starter prompts will only be shown when there is no messages on the chat
                    </span>
                </div>
            </div>

            {/* AI Generation Section */}
            <Box sx={{ mt: 2, mb: 1 }}>
                <SwitchInput
                    label='AI-Powered Starter Prompts'
                    onChange={(value) => setAiEnabled(value)}
                    value={aiEnabled}
                />
            </Box>

            {aiEnabled && (
                <Box
                    sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                        mb: 2,
                        p: 2,
                        borderRadius: 2,
                        border: '1px solid',
                        borderColor: 'divider'
                    }}
                >
                    {!hasFollowUpConfig && (
                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                borderRadius: 10,
                                background: '#fff3cd',
                                padding: 10
                            }}
                        >
                            <span style={{ color: '#856404', fontWeight: 500 }}>
                                Follow-up Prompts must be configured with a provider first. AI Starter Prompts reuses the same LLM
                                provider/credentials.
                            </span>
                        </div>
                    )}

                    {hasFollowUpConfig && (
                        <>
                            <div
                                style={{
                                    display: 'flex',
                                    flexDirection: 'column',
                                    borderRadius: 10,
                                    background: '#e8f4fd',
                                    padding: 10
                                }}
                            >
                                <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                                    <IconSparkles size={20} color='#1976d2' />
                                    <span style={{ color: '#1976d2', marginLeft: 8, fontWeight: 500 }}>
                                        Uses the same LLM provider configured in Follow-up Prompts. When enabled, prompts are generated
                                        dynamically each time the chat opens.
                                    </span>
                                </div>
                            </div>

                            <Typography variant='h5'>Prompt Template</Typography>
                            <Input
                                inputParam={{
                                    label: 'Prompt',
                                    name: 'aiPrompt',
                                    type: 'string',
                                    rows: 4,
                                    description:
                                        'Prompt template for generating starter prompts. Use {context} to insert the chatflow context (system messages, topic content).',
                                    optional: true,
                                    default: defaultAiPrompt
                                }}
                                onChange={(newValue) => setAiPrompt(newValue)}
                                value={aiPrompt}
                            />

                            <StyledButton
                                variant='outlined'
                                onClick={handleGenerate}
                                disabled={isGenerating}
                                startIcon={isGenerating ? <CircularProgress size={16} /> : <IconRefresh size={16} />}
                            >
                                {isGenerating ? 'Generating...' : 'Test Generate'}
                            </StyledButton>

                            {generatedPrompts.length > 0 && (
                                <Box sx={{ mt: 1 }}>
                                    <Typography variant='body2' sx={{ mb: 1, color: 'text.secondary' }}>
                                        Generated prompts (now editable below):
                                    </Typography>
                                    {generatedPrompts.map((prompt, index) => (
                                        <Typography key={index} variant='body2' sx={{ ml: 1 }}>
                                            {index + 1}. {prompt}
                                        </Typography>
                                    ))}
                                </Box>
                            )}
                        </>
                    )}
                </Box>
            )}

            {/* Manual Prompts Section */}
            <Box sx={{ mt: aiEnabled ? 0 : 0 }}>
                {aiEnabled && (
                    <Typography variant='h5' sx={{ mb: 1 }}>
                        Fallback / Manual Prompts
                    </Typography>
                )}
            </Box>
            <Box sx={{ '& > :not(style)': { m: 1 }, pt: aiEnabled ? 0 : 2 }}>
                <List>
                    {inputFields.map((data, index) => {
                        return (
                            <div key={index} style={{ display: 'flex', width: '100%' }}>
                                <Box sx={{ width: '95%', mb: 1 }}>
                                    <OutlinedInput
                                        sx={{ width: '100%' }}
                                        key={index}
                                        type='text'
                                        onChange={(e) => handleChange(index, e)}
                                        size='small'
                                        value={data.prompt}
                                        name='prompt'
                                        endAdornment={
                                            <InputAdornment position='end' sx={{ padding: '2px' }}>
                                                {inputFields.length > 1 && (
                                                    <IconButton
                                                        sx={{ height: 30, width: 30 }}
                                                        size='small'
                                                        color='error'
                                                        disabled={inputFields.length === 1}
                                                        onClick={() => removeInputFields(index)}
                                                        edge='end'
                                                    >
                                                        <IconTrash />
                                                    </IconButton>
                                                )}
                                            </InputAdornment>
                                        }
                                    />
                                </Box>
                                <Box sx={{ width: '5%', mb: 1 }}>
                                    {index === inputFields.length - 1 && (
                                        <IconButton color='primary' onClick={addInputField}>
                                            <IconPlus />
                                        </IconButton>
                                    )}
                                </Box>
                            </div>
                        )
                    })}
                </List>
            </Box>
            <StyledButton variant='contained' onClick={onSave}>
                Save
            </StyledButton>
        </>
    )
}

StarterPrompts.propTypes = {
    dialogProps: PropTypes.object,
    onConfirm: PropTypes.func
}

export default StarterPrompts
