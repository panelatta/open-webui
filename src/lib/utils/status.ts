export function mergeStatus(history: any[] = [], status: any): any[] {
	if (status?.action === 'generation_config' && status?.id) {
		const index = history.findIndex(
			(item) => item?.action === status.action && item?.id === status.id
		);
		if (index !== -1) return history.map((item, i) => (i === index ? status : item));
	}
	return [...history, status];
}
