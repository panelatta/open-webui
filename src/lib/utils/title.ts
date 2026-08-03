const collectResponseText = (response: any): string[] => {
	const message = response?.choices?.[0]?.message ?? {};
	const outputText = Array.isArray(response?.output)
		? response.output.flatMap((item: any) =>
				Array.isArray(item?.content)
					? item.content
							.filter((part: any) => part?.type === 'output_text')
							.map((part: any) => part?.text)
					: []
			)
		: [];

	return [
		message?.content,
		message?.reasoning_content,
		response?.output_text,
		...outputText
	].filter((value): value is string => typeof value === 'string' && value.trim() !== '');
};

const parseJsonTitle = (text: string): string | null => {
	const start = text.indexOf('{');
	const end = text.lastIndexOf('}');
	if (start === -1 || end < start) return null;

	try {
		const parsed = JSON.parse(text.slice(start, end + 1));
		return typeof parsed?.title === 'string' && parsed.title.trim() !== ''
			? parsed.title.trim()
			: null;
	} catch {
		return null;
	}
};

export const parseGeneratedTitle = (response: any): string | null => {
	const candidates = collectResponseText(response);

	for (const candidate of candidates) {
		const title = parseJsonTitle(candidate);
		if (title) return title;
	}

	// Some OpenAI-compatible providers ignore the JSON-only instruction and
	// return a single plain-text title. Accept only a short, single-line value
	// so explanatory responses are not mistaken for titles.
	const plainText = candidates[0]
		?.replace(/^```(?:json)?\s*/i, '')
		.replace(/\s*```$/, '')
		.trim();
	if (plainText && !plainText.includes('\n') && plainText.length <= 200) {
		return plainText.replace(/^["'`]+|["'`]+$/g, '').trim() || null;
	}

	return null;
};
