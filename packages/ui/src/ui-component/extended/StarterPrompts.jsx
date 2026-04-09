import { useDispatch } from 'react-redux'
import { useState, useEffect } from 'react'
import PropTypes from 'prop-types'
import { enqueueSnackbar as enqueueSnackbarAction, closeSnackbar as closeSnackbarAction, SET_CHATFLOW } from '@/store/actions'
import { useTheme } from '@mui/material/styles'

// material-ui
import {
    Button,
    IconButton,
    OutlinedInput,
    Box,
    List,
    InputAdornment,
    Typography,
    Chip,
    FormControl,
    ListItem,
    ListItemAvatar,
    ListItemText,
    MenuItem,
    Select
} from '@mui/material'
import { IconX, IconTrash, IconPlus, IconBulb } from '@tabler/icons-react'

// Project import
import { StyledButton } from '@/ui-component/button/StyledButton'
import { SwitchInput } from '@/ui-component/switch/Switch'
import { Input } from '@/ui-component/input/Input'
import { TooltipWithParser } from '@/ui-component/tooltip/TooltipWithParser'
import { AsyncDropdown } from '@/ui-component/dropdown/AsyncDropdown'
import { Dropdown } from '@/ui-component/dropdown/Dropdown'
import CredentialInputHandler from '@/views/canvas/CredentialInputHandler'
import chatflowsApi from '@/api/chatflows'

// Icons
import anthropicIcon from '@/assets/images/anthropic.svg'
import azureOpenAiIcon from '@/assets/images/azure_openai.svg'
import mistralAiIcon from '@/assets/images/mistralai.svg'
import openAiIcon from '@/assets/images/openai.svg'
import groqIcon from '@/assets/images/groq.png'
import geminiIcon from '@/assets/images/gemini.png'
import ollamaIcon from '@/assets/images/ollama.svg'

// store
import useNotifier from '@/utils/useNotifier'

const starterPromptDescription =
    'Prompt to generate starter questions. Available variable: {context} (chatflow context including system messages and topic content).'
const defaultPrompt =
    "Based on the following context, generate 4 short starter prompts a user might ask when first opening the chat. Each should be concise (under 100 characters), written from the user's perspective, and demonstrate different aspects of what this chatbot can help with.\n\nContext:\n{context}"

const StarterPromptProviders = {
    ANTHROPIC: 'chatAnthropic',
    AZURE_OPENAI: 'azureChatOpenAI',
    GOOGLE_GENAI: 'chatGoogleGenerativeAI',
    GROQ: 'groqChat',
    MISTRALAI: 'chatMistralAI',
    OPENAI: 'chatOpenAI',
    OLLAMA: 'ollama'
}

const starterPromptsOptions = {
    [StarterPromptProviders.ANTHROPIC]: {
        label: 'Anthropic Claude',
        name: StarterPromptProviders.ANTHROPIC,
        icon: anthropicIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['anthropicApi']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.AZURE_OPENAI]: {
        label: 'Azure ChatOpenAI',
        name: StarterPromptProviders.AZURE_OPENAI,
        icon: azureOpenAiIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['azureOpenAIApi']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.GOOGLE_GENAI]: {
        label: 'Google Gemini',
        name: StarterPromptProviders.GOOGLE_GENAI,
        icon: geminiIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['googleGenerativeAI']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.GROQ]: {
        label: 'Groq',
        name: StarterPromptProviders.GROQ,
        icon: groqIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['groqApi']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.MISTRALAI]: {
        label: 'Mistral AI',
        name: StarterPromptProviders.MISTRALAI,
        icon: mistralAiIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['mistralAIApi']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.OPENAI]: {
        label: 'OpenAI',
        name: StarterPromptProviders.OPENAI,
        icon: openAiIcon,
        inputs: [
            {
                label: 'Connect Credential',
                name: 'credential',
                type: 'credential',
                credentialNames: ['openAIApi']
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'asyncOptions',
                loadMethod: 'listModels'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.9
            }
        ]
    },
    [StarterPromptProviders.OLLAMA]: {
        label: 'Ollama',
        name: StarterPromptProviders.OLLAMA,
        icon: ollamaIcon,
        inputs: [
            {
                label: 'Base URL',
                name: 'baseUrl',
                type: 'string',
                placeholder: 'http://127.0.0.1:11434',
                description: 'Base URL of your Ollama instance',
                default: 'http://127.0.0.1:11434'
            },
            {
                label: 'Model Name',
                name: 'modelName',
                type: 'string',
                placeholder: 'llama2',
                description: 'Name of the Ollama model to use',
                default: 'llama3.2-vision:latest'
            },
            {
                label: 'Prompt',
                name: 'prompt',
                type: 'string',
                rows: 4,
                description: starterPromptDescription,
                optional: true,
                default: defaultPrompt
            },
            {
                label: 'Temperature',
                name: 'temperature',
                type: 'number',
                step: 0.1,
                optional: true,
                default: 0.7
            }
        ]
    }
}

const StarterPrompts = ({ dialogProps, onConfirm }) => {
    const dispatch = useDispatch()

    useNotifier()
    const theme = useTheme()

    const enqueueSnackbar = (...args) => dispatch(enqueueSnackbarAction(...args))
    const closeSnackbar = (...args) => dispatch(closeSnackbarAction(...args))

    // Manual prompts state
    const [inputFields, setInputFields] = useState([{ prompt: '' }])

    // AI config state
    const [aiConfig, setAiConfig] = useState({})
    const [selectedProvider, setSelectedProvider] = useState('none')

    const [chatbotConfig, setChatbotConfig] = useState({})

    const addInputField = () => {
        setInputFields([...inputFields, { prompt: '' }])
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

    // AI config handlers
    const handleAiChange = (key, value) => {
        setAiConfig({ ...aiConfig, [key]: value })
    }

    const handleSelectedProviderChange = (event) => {
        const provider = event.target.value
        setSelectedProvider(provider)
        handleAiChange('selectedProvider', provider)
    }

    const setProviderValue = (value, providerName, inputParamName) => {
        let newVal = {}
        if (!Object.prototype.hasOwnProperty.call(aiConfig, providerName)) {
            newVal = { ...aiConfig, [providerName]: {} }
        } else {
            newVal = { ...aiConfig }
        }
        newVal[providerName][inputParamName] = value
        setAiConfig(newVal)
        return newVal
    }

    const checkDisabled = () => {
        if (aiConfig && aiConfig.status) {
            if (selectedProvider === 'none') return true
            const provider = starterPromptsOptions[selectedProvider]
            if (!provider) return true
            for (let inputParam of provider.inputs) {
                if (!inputParam.optional) {
                    const param = inputParam.name === 'credential' ? 'credentialId' : inputParam.name
                    if (!aiConfig[selectedProvider] || !aiConfig[selectedProvider][param] || aiConfig[selectedProvider][param] === '') {
                        return true
                    }
                }
            }
        }
        return false
    }

    const onSave = async () => {
        try {
            // Build starterPrompts with manual prompts + aiConfig
            let starterPrompts = { ...inputFields }

            // If AI is enabled and prompt is not set, save default
            if (aiConfig.status && selectedProvider && aiConfig[selectedProvider] && starterPromptsOptions[selectedProvider]) {
                if (!aiConfig[selectedProvider].prompt) {
                    aiConfig[selectedProvider].prompt = starterPromptsOptions[selectedProvider].inputs.find(
                        (input) => input.name === 'prompt'
                    )?.default
                }
                if (!aiConfig[selectedProvider].temperature) {
                    aiConfig[selectedProvider].temperature = starterPromptsOptions[selectedProvider].inputs.find(
                        (input) => input.name === 'temperature'
                    )?.default
                }
            }

            starterPrompts.aiConfig = aiConfig
            chatbotConfig.starterPrompts = starterPrompts

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
        if (dialogProps.chatflow && dialogProps.chatflow.chatbotConfig) {
            try {
                let config = JSON.parse(dialogProps.chatflow.chatbotConfig)
                setChatbotConfig(config || {})
                if (config.starterPrompts) {
                    // Load manual prompts
                    let fields = []
                    Object.getOwnPropertyNames(config.starterPrompts).forEach((key) => {
                        if (key !== 'aiConfig' && config.starterPrompts[key]) {
                            fields.push(config.starterPrompts[key])
                        }
                    })
                    if (fields.length > 0) setInputFields(fields)
                    else setInputFields([{ prompt: '' }])

                    // Load AI config
                    if (config.starterPrompts.aiConfig) {
                        setAiConfig(config.starterPrompts.aiConfig)
                        setSelectedProvider(config.starterPrompts.aiConfig.selectedProvider || 'none')
                    }
                } else {
                    setInputFields([{ prompt: '' }])
                }
            } catch (e) {
                setInputFields([{ prompt: '' }])
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
                    padding: 10,
                    marginBottom: 16
                }}
            >
                <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                    <IconBulb size={30} color='#2d6a4f' />
                    <span style={{ color: '#2d6a4f', marginLeft: 10, fontWeight: 500 }}>
                        Starter prompts will only be shown when there is no messages on the chat
                    </span>
                </div>
            </div>

            {/* AI-Powered Section */}
            <Box
                sx={{
                    width: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'start',
                    justifyContent: 'start',
                    gap: 3,
                    mb: 2
                }}
            >
                <SwitchInput
                    label='AI-Powered Starter Prompts'
                    onChange={(value) => handleAiChange('status', value)}
                    value={aiConfig.status}
                />
                {aiConfig && aiConfig.status && (
                    <>
                        <Typography variant='h5'>Providers</Typography>
                        <FormControl fullWidth>
                            <Select
                                size='small'
                                value={selectedProvider}
                                onChange={handleSelectedProviderChange}
                                sx={{
                                    '& .MuiSvgIcon-root': {
                                        color: theme?.customization?.isDarkMode ? '#fff' : 'inherit'
                                    }
                                }}
                            >
                                {Object.values(starterPromptsOptions).map((provider) => (
                                    <MenuItem key={provider.name} value={provider.name}>
                                        {provider.label}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                        {selectedProvider !== 'none' && (
                            <>
                                <ListItem sx={{ p: 0 }} alignItems='center'>
                                    <ListItemAvatar>
                                        <div
                                            style={{
                                                width: 50,
                                                height: 50,
                                                borderRadius: '50%',
                                                backgroundColor: 'white'
                                            }}
                                        >
                                            <img
                                                style={{
                                                    width: '100%',
                                                    height: '100%',
                                                    padding: 10,
                                                    objectFit: 'contain'
                                                }}
                                                alt='AI'
                                                src={starterPromptsOptions[selectedProvider].icon}
                                            />
                                        </div>
                                    </ListItemAvatar>
                                    <ListItemText primary={starterPromptsOptions[selectedProvider].label} />
                                </ListItem>
                                {starterPromptsOptions[selectedProvider].inputs.map((inputParam, index) => (
                                    <Box key={index} sx={{ px: 2, width: '100%' }}>
                                        <div style={{ display: 'flex', flexDirection: 'row' }}>
                                            <Typography>
                                                {inputParam.label}
                                                {!inputParam.optional && <span style={{ color: 'red' }}>&nbsp;*</span>}
                                                {inputParam.description && (
                                                    <TooltipWithParser style={{ marginLeft: 10 }} title={inputParam.description} />
                                                )}
                                            </Typography>
                                        </div>
                                        {inputParam.type === 'credential' && (
                                            <CredentialInputHandler
                                                key={`${selectedProvider}-${inputParam.name}`}
                                                data={
                                                    aiConfig[selectedProvider]?.credentialId
                                                        ? { credential: aiConfig[selectedProvider].credentialId }
                                                        : {}
                                                }
                                                inputParam={inputParam}
                                                onSelect={(newValue) => setProviderValue(newValue, selectedProvider, 'credentialId')}
                                            />
                                        )}

                                        {(inputParam.type === 'string' ||
                                            inputParam.type === 'password' ||
                                            inputParam.type === 'number') && (
                                            <>
                                                <Input
                                                    key={`${selectedProvider}-${inputParam.name}`}
                                                    inputParam={inputParam}
                                                    onChange={(newValue) =>
                                                        setProviderValue(newValue, selectedProvider, inputParam.name)
                                                    }
                                                    value={
                                                        aiConfig[selectedProvider] && aiConfig[selectedProvider][inputParam.name]
                                                            ? aiConfig[selectedProvider][inputParam.name]
                                                            : inputParam.default ?? ''
                                                    }
                                                />
                                                {inputParam.name === 'prompt' && (
                                                    <Box sx={{ display: 'flex', gap: 0.5, mt: 1, flexWrap: 'wrap' }}>
                                                        {['{context}'].map((variable) => (
                                                            <Chip
                                                                key={variable}
                                                                label={variable}
                                                                size='small'
                                                                variant='outlined'
                                                                onClick={() => {
                                                                    const current =
                                                                        aiConfig[selectedProvider]?.[inputParam.name] ||
                                                                        inputParam.default ||
                                                                        ''
                                                                    setProviderValue(
                                                                        current + ' ' + variable,
                                                                        selectedProvider,
                                                                        inputParam.name
                                                                    )
                                                                }}
                                                            />
                                                        ))}
                                                    </Box>
                                                )}
                                            </>
                                        )}

                                        {inputParam.type === 'asyncOptions' && (
                                            <div style={{ display: 'flex', flexDirection: 'row' }}>
                                                <AsyncDropdown
                                                    key={`${selectedProvider}-${inputParam.name}`}
                                                    name={inputParam.name}
                                                    nodeData={{
                                                        name: starterPromptsOptions[selectedProvider].name,
                                                        inputParams: starterPromptsOptions[selectedProvider].inputs
                                                    }}
                                                    value={
                                                        aiConfig[selectedProvider] && aiConfig[selectedProvider][inputParam.name]
                                                            ? aiConfig[selectedProvider][inputParam.name]
                                                            : inputParam.default ?? 'choose an option'
                                                    }
                                                    onSelect={(newValue) =>
                                                        setProviderValue(newValue, selectedProvider, inputParam.name)
                                                    }
                                                />
                                            </div>
                                        )}

                                        {inputParam.type === 'options' && (
                                            <Dropdown
                                                name={inputParam.name}
                                                options={inputParam.options}
                                                onSelect={(newValue) => setProviderValue(newValue, selectedProvider, inputParam.name)}
                                                value={
                                                    aiConfig[selectedProvider] && aiConfig[selectedProvider][inputParam.name]
                                                        ? aiConfig[selectedProvider][inputParam.name]
                                                        : inputParam.default ?? 'choose an option'
                                                }
                                            />
                                        )}
                                    </Box>
                                ))}
                            </>
                        )}
                    </>
                )}
            </Box>

            {/* Manual / Fallback Prompts */}
            {aiConfig.status && (
                <Typography variant='h5' sx={{ mt: 2 }}>
                    Fallback / Manual Prompts
                </Typography>
            )}
            <Box sx={{ '& > :not(style)': { m: 1 }, pt: 2 }}>
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
            <StyledButton disabled={checkDisabled()} variant='contained' onClick={onSave}>
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
