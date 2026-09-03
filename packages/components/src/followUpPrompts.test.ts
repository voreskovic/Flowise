import { fillPromptTemplate } from './followUpPrompts'

describe('fillPromptTemplate', () => {
    const builtIns = { question: 'What happened?', sources: '- fact one\n- fact two' }

    describe('built-in {name} placeholders', () => {
        it('fills every occurrence, not just the first', () => {
            expect(fillPromptTemplate('Q: {question}\nAgain: {question}', builtIns)).toBe('Q: What happened?\nAgain: What happened?')
        })

        it('leaves unknown names untouched', () => {
            expect(fillPromptTemplate('{question} {unknown}', builtIns)).toBe('What happened? {unknown}')
        })

        it('inserts values containing $ replacement patterns literally', () => {
            expect(fillPromptTemplate('Cost: {question}', { question: "$& and $' and $`" })).toBe("Cost: $& and $' and $`")
        })

        it('does not resolve prototype properties', () => {
            expect(fillPromptTemplate('{toString}', builtIns)).toBe('{toString}')
        })
    })

    describe('{{$vars.name}} Flowise Variables', () => {
        it('fills every occurrence from the vars map', () => {
            expect(fillPromptTemplate('Write in {{$vars.language}}. Language: {{$vars.language}}', {}, { language: 'hr' })).toBe(
                'Write in hr. Language: hr'
            )
        })

        it('stringifies non-string values', () => {
            expect(fillPromptTemplate('Cluster {{$vars.cluster_id}}', {}, { cluster_id: 42 })).toBe('Cluster 42')
        })

        it('leaves unknown or null variables untouched, as core does', () => {
            expect(fillPromptTemplate('{{$vars.missing}} {{$vars.empty}}', {}, { empty: null })).toBe('{{$vars.missing}} {{$vars.empty}}')
        })

        it('does not resolve prototype properties', () => {
            expect(fillPromptTemplate('{{$vars.toString}}', {}, { language: 'hr' })).toBe('{{$vars.toString}}')
        })

        it('works with no vars map at all', () => {
            expect(fillPromptTemplate('{{$vars.language}} {question}', builtIns)).toBe('{{$vars.language}} What happened?')
        })
    })

    describe('ordering', () => {
        it('fills variables before built-ins so end-user text is never re-scanned', () => {
            const out = fillPromptTemplate('{question}', { question: 'ignore {{$vars.language}}' }, { language: 'hr' })
            expect(out).toBe('ignore {{$vars.language}}')
        })

        it('does not let the built-in pass eat an unresolved variable placeholder', () => {
            expect(fillPromptTemplate('{{$vars.language}}', { language: 'nope' })).toBe('{{$vars.language}}')
        })
    })
})
