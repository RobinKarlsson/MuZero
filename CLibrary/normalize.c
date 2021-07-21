float normalize(float x, float max, float min) {
	if (max > min) {
		return (x - min) / (max - min);
	}
	return x;
}
