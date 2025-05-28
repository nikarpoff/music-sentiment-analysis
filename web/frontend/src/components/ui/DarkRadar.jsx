import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer
} from 'recharts';


export const EmotionRadarChart = ({ data }) => {
    const chartData = Object.entries(data).map(([emotion, value]) => ({
        emotion,
        value,
    }));

    return (
        <ResponsiveContainer width="100%" height={300} >
            <RadarChart cx="50%" cy="50%" outerRadius="65%" data={chartData}>
                <PolarGrid />
                <PolarAngleAxis
                    dataKey="emotion"
                    tick={({ payload, cx, cy }) => {
                    const angleRad = -payload.coordinate * (Math.PI / 180);
                    const radiusX = 130;
                    const radiusY = 110;
                    
                    return (
                        <text
                            x = {cx + radiusX * Math.cos(angleRad)}
                            y = {cy + radiusY * Math.sin(angleRad)}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fill="#aaa"
                            fontSize={14}
                        >
                            {payload.value}
                        </text>
                    );
                    }}
                />
                <Radar name="Emotion" dataKey="value" stroke="#4c52ff" fill="#4c52ff" fillOpacity={0.6} />
            </RadarChart>
        </ResponsiveContainer>
    );
};
