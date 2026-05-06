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
    Checkbox,
    FormControl,
    FormControlLabel,
    ListItem,
    ListItemAvatar,
    ListItemText,
    MenuItem,
    Select
} from '@mui/material'
import { IconX, IconTrash, IconPlus, IconBulb, IconDatabase } from '@tabler/icons-react'

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
    'Prompt to generate starter questions. Available variables: {context} (chatflow context including system messages and topic content). When "Retrieve from Vector DB" is enabled, also: {retrieved_from_vector_db} — fetches article texts by ID from overrideConfig.qdrantFilter.must[].has_id and feeds them to the LLM. Including this variable disables the cache shortcut.'
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

const QdrantSection = ({ title, description, state, handlers, theme, allowMetadataOnly }) => {
    const { handleChange, addMetadataField, removeMetadataField, updateMetadataField } = handlers

    return (
        <Box
            sx={{
                width: '100%',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 2,
                p: 2,
                mb: 2,
                display: 'flex',
                flexDirection: 'column',
                gap: 2
            }}
        >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <IconDatabase size={20} />
                <Typography variant='h5'>{title}</Typography>
            </Box>
            {description && (
                <Typography variant='body2' sx={{ color: 'text.secondary', mt: -1 }}>
                    {description}
                </Typography>
            )}

            <SwitchInput label={`Enable ${title}`} onChange={(value) => handleChange('enabled', value)} value={state.enabled} />

            {state.enabled && allowMetadataOnly && (
                <Box sx={{ pl: 0 }}>
                    <SwitchInput
                        label='Metadata-only lookup (skip vector similarity)'
                        onChange={(value) => handleChange('metadataOnly', value)}
                        value={!!state.metadataOnly}
                    />
                    <Typography variant='body2' sx={{ color: 'text.secondary', mt: -1 }}>
                        When on, retrieve uses only the metadata filter below (no embedding call). Returns the first matching points
                        as cached starter prompts; if none match, falls through to LLM generation.
                    </Typography>
                </Box>
            )}

            {state.enabled && (
                <>
                    {/* Connection */}
                    <Typography variant='h5' sx={{ mt: 1 }}>
                        Connection
                    </Typography>
                    <Box sx={{ px: 2, width: '100%' }}>
                        <Typography>
                            Qdrant Server URL <span style={{ color: 'red' }}>*</span>
                        </Typography>
                        <Input
                            inputParam={{
                                label: 'Qdrant Server URL',
                                name: 'qdrantServerUrl',
                                type: 'string',
                                placeholder: 'https://your-qdrant-instance.qdrant.io'
                            }}
                            onChange={(newValue) => handleChange('qdrantServerUrl', newValue)}
                            value={state.qdrantServerUrl || ''}
                        />
                    </Box>
                    <Box sx={{ px: 2, width: '100%' }}>
                        <Typography>Qdrant API Key</Typography>
                        <CredentialInputHandler
                            data={state.qdrantCredentialId ? { credential: state.qdrantCredentialId } : {}}
                            inputParam={{
                                label: 'Connect Credential',
                                name: 'credential',
                                type: 'credential',
                                credentialNames: ['qdrantApi']
                            }}
                            onSelect={(newValue) => handleChange('qdrantCredentialId', newValue)}
                        />
                    </Box>
                    <Box sx={{ px: 2, width: '100%' }}>
                        <Typography>
                            Collection Name <span style={{ color: 'red' }}>*</span>
                        </Typography>
                        <Input
                            inputParam={{
                                label: 'Collection Name',
                                name: 'collectionName',
                                type: 'string',
                                placeholder: 'starter_prompts'
                            }}
                            onChange={(newValue) => handleChange('collectionName', newValue)}
                            value={state.collectionName || ''}
                        />
                    </Box>
                    <Box sx={{ px: 2, width: '100%' }}>
                        <Typography>Vector Dimension</Typography>
                        <Input
                            inputParam={{
                                label: 'Vector Dimension',
                                name: 'vectorDimension',
                                type: 'number',
                                default: 1536
                            }}
                            onChange={(newValue) => handleChange('vectorDimension', parseInt(newValue) || 1536)}
                            value={state.vectorDimension || 1536}
                        />
                    </Box>

                    {/* OpenAI Embeddings — only used by the vector-similarity path */}
                    {!state.metadataOnly && (
                        <>
                            <Typography variant='h5' sx={{ mt: 1 }}>
                                OpenAI Embeddings
                            </Typography>
                            <Box sx={{ px: 2, width: '100%' }}>
                                <Typography>
                                    Connect Credential <span style={{ color: 'red' }}>*</span>
                                </Typography>
                                <CredentialInputHandler
                                    data={state.embeddingCredentialId ? { credential: state.embeddingCredentialId } : {}}
                                    inputParam={{
                                        label: 'Connect Credential',
                                        name: 'credential',
                                        type: 'credential',
                                        credentialNames: ['openAIApi']
                                    }}
                                    onSelect={(newValue) => handleChange('embeddingCredentialId', newValue)}
                                />
                            </Box>
                            <Box sx={{ px: 2, width: '100%' }}>
                                <Typography>Model Name</Typography>
                                <Input
                                    inputParam={{
                                        label: 'Model Name',
                                        name: 'embeddingModelName',
                                        type: 'string',
                                        placeholder: 'text-embedding-3-small',
                                        optional: true
                                    }}
                                    onChange={(newValue) => handleChange('embeddingModelName', newValue)}
                                    value={state.embeddingModelName || ''}
                                />
                            </Box>
                            <Box sx={{ px: 2, width: '100%' }}>
                                <Typography>
                                    Base Path
                                    <TooltipWithParser
                                        style={{ marginLeft: 10 }}
                                        title='Optional custom base URL for the OpenAI API (e.g., for Azure-compatible endpoints)'
                                    />
                                </Typography>
                                <Input
                                    inputParam={{
                                        label: 'Base Path',
                                        name: 'embeddingBasePath',
                                        type: 'string',
                                        optional: true,
                                        placeholder: 'https://api.example.com/v1'
                                    }}
                                    onChange={(newValue) => handleChange('embeddingBasePath', newValue)}
                                    value={state.embeddingBasePath || ''}
                                />
                            </Box>
                        </>
                    )}

                    {/* Metadata Fields */}
                    <Typography variant='h5' sx={{ mt: 1 }}>
                        Metadata Fields
                    </Typography>
                    <Typography variant='body2' sx={{ px: 2, color: 'text.secondary' }}>
                        Source Var maps to <code>overrideConfig.vars.&lt;name&gt;</code> at runtime.
                    </Typography>
                    <Box sx={{ px: 2, width: '100%' }}>
                        {(state.metadataFields || []).map((field, index) => (
                            <Box key={index} sx={{ display: 'flex', gap: 1, mb: 1, alignItems: 'center', flexWrap: 'wrap' }}>
                                <OutlinedInput
                                    size='small'
                                    placeholder='Key'
                                    value={field.key}
                                    onChange={(e) => updateMetadataField(index, 'key', e.target.value)}
                                    sx={{ flex: 1, minWidth: 100 }}
                                />
                                <OutlinedInput
                                    size='small'
                                    placeholder='Source Var'
                                    value={field.sourceVar}
                                    onChange={(e) => updateMetadataField(index, 'sourceVar', e.target.value)}
                                    sx={{ flex: 1, minWidth: 100 }}
                                />
                                <Select
                                    size='small'
                                    value={field.type || 'string'}
                                    onChange={(e) => updateMetadataField(index, 'type', e.target.value)}
                                    sx={{
                                        minWidth: 90,
                                        '& .MuiSvgIcon-root': { color: theme?.customization?.isDarkMode ? '#fff' : 'inherit' }
                                    }}
                                >
                                    <MenuItem value='string'>string</MenuItem>
                                    <MenuItem value='number'>number</MenuItem>
                                    <MenuItem value='array'>array</MenuItem>
                                </Select>
                                <FormControlLabel
                                    control={
                                        <Checkbox
                                            size='small'
                                            checked={field.useAsFilter ?? true}
                                            onChange={(e) => updateMetadataField(index, 'useAsFilter', e.target.checked)}
                                        />
                                    }
                                    label='Filter'
                                    sx={{ mr: 0 }}
                                />
                                <IconButton size='small' color='error' onClick={() => removeMetadataField(index)}>
                                    <IconTrash size={16} />
                                </IconButton>
                            </Box>
                        ))}
                        <Button size='small' startIcon={<IconPlus size={14} />} onClick={addMetadataField} sx={{ mt: 0.5 }}>
                            Add Field
                        </Button>
                    </Box>
                </>
            )}
        </Box>
    )
}

QdrantSection.propTypes = {
    title: PropTypes.string,
    description: PropTypes.string,
    state: PropTypes.object,
    handlers: PropTypes.object,
    theme: PropTypes.object,
    allowMetadataOnly: PropTypes.bool
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

    // Qdrant config state — separate for retrieve and store
    const defaultQdrantSection = {
        enabled: false,
        metadataOnly: false,
        qdrantServerUrl: '',
        qdrantCredentialId: '',
        collectionName: '',
        vectorDimension: 1536,
        embeddingCredentialId: '',
        embeddingModelName: '',
        embeddingBasePath: '',
        metadataFields: []
    }
    const [qdrantRetrieve, setQdrantRetrieve] = useState({ ...defaultQdrantSection })
    const [qdrantStore, setQdrantStore] = useState({ ...defaultQdrantSection })

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

    // Generic Qdrant section handlers (works for both retrieve and store)
    const makeQdrantHandlers = (state, setState) => ({
        handleChange: (key, value) => setState({ ...state, [key]: value }),
        addMetadataField: () =>
            setState({
                ...state,
                metadataFields: [...(state.metadataFields || []), { key: '', sourceVar: '', type: 'string', useAsFilter: true }]
            }),
        removeMetadataField: (index) => {
            const fields = [...(state.metadataFields || [])]
            fields.splice(index, 1)
            setState({ ...state, metadataFields: fields })
        },
        updateMetadataField: (index, fieldKey, value) => {
            const fields = [...(state.metadataFields || [])]
            fields[index] = { ...fields[index], [fieldKey]: value }
            setState({ ...state, metadataFields: fields })
        }
    })

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
            starterPrompts.qdrantConfig = {
                retrieve: qdrantRetrieve,
                store: qdrantStore
            }
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
                        if (key !== 'aiConfig' && key !== 'qdrantConfig' && config.starterPrompts[key]) {
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

                    // Load Qdrant configs
                    if (config.starterPrompts.qdrantConfig) {
                        const qc = config.starterPrompts.qdrantConfig
                        if (qc.retrieve) setQdrantRetrieve({ ...defaultQdrantSection, ...qc.retrieve })
                        if (qc.store) setQdrantStore({ ...defaultQdrantSection, ...qc.store })
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
                                                        {[
                                                            '{context}',
                                                            ...(qdrantRetrieve?.enabled ? ['{retrieved_from_vector_db}'] : [])
                                                        ].map((variable) => (
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

            {/* Qdrant: Retrieve from Vector DB */}
            {aiConfig && aiConfig.status && (
                <QdrantSection
                    title='Retrieve from Vector DB'
                    description='Query Qdrant for cached starter prompts before calling the LLM. Filters by metadata from overrideConfig.vars.'
                    state={qdrantRetrieve}
                    handlers={makeQdrantHandlers(qdrantRetrieve, setQdrantRetrieve)}
                    theme={theme}
                    allowMetadataOnly
                />
            )}

            {/* Qdrant: Store to Vector DB */}
            {aiConfig && aiConfig.status && (
                <QdrantSection
                    title='Store to Vector DB'
                    description='After generating prompts via LLM, store each prompt as a vector in Qdrant for future retrieval.'
                    state={qdrantStore}
                    handlers={makeQdrantHandlers(qdrantStore, setQdrantStore)}
                    theme={theme}
                />
            )}

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
