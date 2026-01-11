interface AttentionStatsProps {
  needWater: number;
  lowLight: number;
  darkMode: boolean;
}

export function AttentionStats({ needWater, lowLight, darkMode }: AttentionStatsProps) {
  return (
    <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} space-y-1`}>
      <div>{needWater} {needWater === 1 ? 'plant needs' : 'plants need'} water</div>
      <div>{lowLight} {lowLight === 1 ? 'plant needs' : 'plants need'} better light</div>
    </div>
  );
}
