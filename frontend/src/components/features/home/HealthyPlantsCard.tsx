import Image from 'next/image';

interface HealthyPlantsCardProps {
  percentage: number;
  darkMode: boolean;
}

export function HealthyPlantsCard({ percentage, darkMode }: HealthyPlantsCardProps) {
  return (
    <div className={`p-5 rounded-3xl ${darkMode ? 'bg-accent' : 'bg-accent'} shadow-md flex flex-col items-center justify-center`}>
      <Image src="/ic-5.png" alt="Healthy" width={36} height={36} className="mb-2" />
      <p className="text-3xl font-bold text-white">{percentage}%</p>
      <p className="text-xs text-white/80 text-center">healthy</p>
    </div>
  );
}
